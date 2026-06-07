/**
 * Builds a context block from memory for injection into the system prompt.
 *
 * Two modes:
 * - Selective (prompt provided): search semantic memory for entries relevant
 *   to the user's current prompt, plus always-inject lessons.
 * - Fallback (no prompt): dump top entries by prefix (old behavior).
 */
import type { MemoryStore, SemanticEntry, LessonEntry } from "./store.js";
import os from "node:os";

const MAX_CONTEXT_CHARS = 8000;
const SEARCH_LIMIT = 15;
const LESSON_SEARCH_LIMIT = 15;

export interface ContextBlock {
  text: string;
  stats: { semantic: number; lessons: number };
}

/**
 * Context about recent agent tool activity, collected between turns.
 * Used to enrich memory search queries beyond the user's prompt alone.
 */
export interface ToolTurnContext {
  /** Names of tools called since the last agent turn */
  toolNames: string[];
  /** The most recently called tool */
  lastTool?: string;
  /** Whether any tool call produced an error */
  hasErrors: boolean;
  /** Error-related terms extracted from failed tool calls */
  errorTerms: string[];
  /** Whether edit/write tools were called (triggers tool.* prefix search) */
  editActivity: boolean;
  /** Whether test-related commands were run (triggers debugging/testing lesson search) */
  testActivity: boolean;
}

/**
 * Configuration for lesson injection behavior.
 * - "all": inject all lessons (original behavior, default)
 * - "selective": use semantic search to pick relevant lessons + category filtering
 */
export type LessonInjectionMode = "all" | "selective";

export interface InjectorConfig {
  lessonInjection?: LessonInjectionMode;
  /**
   * Opt-in: restore per-user-message selective injection.
   *
   * When false (default), pi-memory injects a one-shot fallback block at
   * session_start (correct message ordering, stable prefix cache).
   *
   * When true, the session_start dump is skipped and each turn runs a
   * semantic search against the user's current prompt; the result is
   * appended to `event.systemPrompt` in `before_agent_start`.
   *
   * Tradeoffs:
   * - Pro: per-query relevance — facts outside the 8KB fallback dump reach
   *   the model when they match the current prompt.
   * - Con: the system prompt mutates every turn, invalidating the provider's
   *   prefix cache after the system block (Bedrock / Anthropic cache_control).
   *   Conversation suffix gets re-written at cacheWrite rates on every user
   *   turn boundary (~12.5x cacheRead on Claude).
   *
   * Correctness is preserved either way: systemPrompt is a separate field
   * from the messages list, so the user's question remains the last
   * user-role message and the model responds to it.
   */
  perTurnInjection?: boolean;
  /**
   * Model string passed to `pi --model` for session-end consolidation.
   * When omitted, the built-in default is used.  Useful for users on
   * non-Anthropic providers (OpenAI/Codex/OpenRouter/Ollama/local),
   * or for picking a cheaper/faster model for background extraction.
   *
   * Invalid model strings will cause the consolidation sub-process to
   * fail — the existing try/catch swallows that silently, so the worst
   * case is that consolidation skips this session.
   */
  consolidationModel?: string;
}

/**
 * Build context block. When `prompt` is provided, uses selective injection
 * (search-based). Otherwise falls back to prefix-based dump.
 */
export function buildContextBlock(store: MemoryStore, cwd?: string, prompt?: string, config?: InjectorConfig, toolContext?: ToolTurnContext): ContextBlock {
  if (prompt?.trim()) {
    return buildSelectiveBlock(store, prompt, cwd, config, toolContext);
  }
  return buildFallbackBlock(store, cwd);
}

// ─── Selective injection ─────────────────────────────────────────────

function buildSelectiveBlock(store: MemoryStore, prompt: string, cwd?: string, config?: InjectorConfig, toolContext?: ToolTurnContext): ContextBlock {
  const sections: string[] = [];
  let semanticCount = 0;
  let lessonCount = 0;
  const mode = config?.lessonInjection ?? "all";

  // Enrich search query with recent tool activity context
  const queryParts = [prompt];
  if (toolContext) {
    if (toolContext.toolNames.length > 0) queryParts.push(...toolContext.toolNames);
    if (toolContext.errorTerms.length > 0) queryParts.push(...toolContext.errorTerms);
    if (toolContext.lastTool) queryParts.push(toolContext.lastTool);
  }
  const enrichedQuery = queryParts.join(" ");

  const results = store.searchSemantic(enrichedQuery, SEARCH_LIMIT);

  // Also search with project slug if we have a cwd, to pull in project context
  const slug = cwd ? projectSlug(cwd) : "";
  if (slug) {
    const projectResults = store.searchSemantic(slug, 5);
    // Merge, dedup by key
    const seen = new Set(results.map(r => r.key));
    for (const r of projectResults) {
      if (!seen.has(r.key)) {
        results.push(r);
        seen.add(r.key);
      }
    }
  }

  // Filter out project.* facts that belong to other projects.
  // FTS can match project facts from unrelated projects when their text
  // coincidentally matches the prompt (e.g. a prompt about "prisma" pulling
  // in rise.testing.fabricca). Keep only facts whose project segment
  // matches the current slug; non-project facts (pref.*, tool.*, user.*)
  // are always kept.
  const filteredResults = slug
    ? results.filter(r => {
        if (!r.key.startsWith("project.")) return true;
        const parts = r.key.split(".");
        return parts.length >= 2 && parts[1] === slug;
      })
    : results;

  if (filteredResults.length > 0) {
    sections.push(formatSection("Relevant Memory", filteredResults.map(formatSemantic)));
    semanticCount = filteredResults.length;

    // Track access time for these memories
    store.touchAccessed(filteredResults.map(r => r.key));
  }

  // ── Activity-triggered bonus searches ──
  // If the agent was editing files, pull in tool.* preferences
  if (toolContext?.editActivity) {
    const toolPrefs = store.listSemantic("tool.", 5);
    for (const tp of toolPrefs) {
      if (!results.some(r => r.key === tp.key)) {
        results.push(tp);
      }
    }
  }

  // If tools errored or tests ran, pull in debugging/testing lessons
  let bonusLessons: LessonEntry[] = [];
  if (toolContext?.hasErrors || toolContext?.testActivity) {
    const searchTerms: string[] = [];
    if (toolContext.hasErrors) {
      searchTerms.push("error", "fail", "debug");
      if (toolContext.lastTool) searchTerms.push(toolContext.lastTool);
      if (toolContext.errorTerms.length > 0) searchTerms.push(...toolContext.errorTerms);
    }
    if (toolContext.testActivity) {
      searchTerms.push("test", "testing");
    }
    if (searchTerms.length > 0) {
      bonusLessons = store.searchLessons(searchTerms.join(" "), 5);
    }
  }

  // ── Lesson injection ──
  const lessons = mode === "selective"
    ? getRelevantLessons(store, prompt, cwd)
    : store.listLessons(undefined, 50, slug || undefined);

  // Merge bonus lessons into the main lesson set (dedup by id)
  const allLessons = [...lessons];
  for (const bl of bonusLessons) {
    if (!allLessons.some(l => l.id === bl.id)) {
      allLessons.push(bl);
    }
  }

  if (allLessons.length > 0) {
    const corrections = allLessons.filter(l => l.negative);
    const positives = allLessons.filter(l => !l.negative);

    if (corrections.length > 0) {
      const formatted = corrections.map(l =>
        `DON'T: ${l.rule}${l.category !== "general" ? ` [${l.category}]` : ""}`
      );
      sections.push(formatSection("Learned Corrections", formatted));
    }
    if (positives.length > 0) {
      const formatted = positives.map(l =>
        `${l.rule}${l.category !== "general" ? ` [${l.category}]` : ""}`
      );
      sections.push(formatSection("Validated Approaches", formatted));
    }
    lessonCount = allLessons.length;
  }

  if (sections.length === 0) {
    return { text: "", stats: { semantic: 0, lessons: 0 } };
  }

  let text = `<memory>\n${sections.join("\n")}\n\n${MEMORY_DRIFT_CAVEAT}\n</memory>`;

  if (text.length > MAX_CONTEXT_CHARS) {
    text = text.slice(0, MAX_CONTEXT_CHARS - 20) + "\n... (truncated)\n</memory>";
  }

  return { text, stats: { semantic: semanticCount, lessons: lessonCount } };
}

// ─── Selective lesson injection ──────────────────────────────────────

/**
 * Get lessons relevant to the current prompt + project context.
 *
 * Strategy:
 * 1. Search lessons by prompt terms (semantic/FTS match)
 * 2. If cwd implies a project, also search by project slug
 * 3. Always include "general" category lessons (broadly applicable)
 * 4. Dedup and cap at LESSON_SEARCH_LIMIT
 */
function getRelevantLessons(store: MemoryStore, prompt: string, cwd?: string): LessonEntry[] {
  const seen = new Set<string>();
  const result: LessonEntry[] = [];

  function add(lessons: LessonEntry[]) {
    for (const l of lessons) {
      if (!seen.has(l.id)) {
        seen.add(l.id);
        result.push(l);
      }
    }
  }

  // 1. Search by prompt relevance (FTS across rule text + category)
  add(store.searchLessons(prompt, LESSON_SEARCH_LIMIT));

  // 2. Search by project slug if we have a cwd
  const slug = cwd ? projectSlug(cwd) : "";
  if (slug) {
    add(store.searchLessons(slug, 5));
  }

  // 3. Always include general lessons (they're broadly applicable)
  add(store.listLessons("general", 10));

  return result.slice(0, LESSON_SEARCH_LIMIT);
}

// ─── Fallback (no prompt) ────────────────────────────────────────────

function buildFallbackBlock(store: MemoryStore, cwd?: string): ContextBlock {
  const sections: string[] = [];
  let semanticCount = 0;
  let lessonCount = 0;

  const prefs = store.listSemantic("pref.", 50);
  if (prefs.length > 0) {
    sections.push(formatSection("User Preferences", prefs.map(formatSemantic)));
    semanticCount += prefs.length;
  }

  const projects = store.listSemantic("project.", 50);
  // Filter project facts to the current project only.
  // Match by exact second key segment (project.<slug>.<rest>) rather than
  // substring — avoids "pi" matching "project.pipefittingjobs.*" and prevents
  // user-set facts (confidence 0.95) from bleeding into unrelated sessions.
  const slug = cwd ? projectSlug(cwd) : "";
  const relevant = slug
    ? projects.filter(p => {
        const parts = p.key.split(".");
        return parts.length >= 2 && parts[1] === slug;
      })
    : projects;
  if (relevant.length > 0) {
    sections.push(formatSection("Project Context", relevant.map(formatSemantic)));
    semanticCount += relevant.length;
  }

  const tools = store.listSemantic("tool.", 20);
  if (tools.length > 0) {
    sections.push(formatSection("Tool Preferences", tools.map(formatSemantic)));
    semanticCount += tools.length;
  }

  const lessons = store.listLessons(undefined, 50, slug || undefined);
  if (lessons.length > 0) {
    const corrections = lessons.filter(l => l.negative);
    const positives = lessons.filter(l => !l.negative);

    if (corrections.length > 0) {
      const formatted = corrections.map(l =>
        `DON'T: ${l.rule}${l.category !== "general" ? ` [${l.category}]` : ""}`
      );
      sections.push(formatSection("Learned Corrections", formatted));
    }
    if (positives.length > 0) {
      const formatted = positives.map(l =>
        `${l.rule}${l.category !== "general" ? ` [${l.category}]` : ""}`
      );
      sections.push(formatSection("Validated Approaches", formatted));
    }
    lessonCount = lessons.length;
  }

  const user = store.listSemantic("user.", 10);
  if (user.length > 0) {
    sections.push(formatSection("User", user.map(formatSemantic)));
    semanticCount += user.length;
  }

  if (sections.length === 0) {
    return { text: "", stats: { semantic: 0, lessons: 0 } };
  }

  let text = `<memory>\n${sections.join("\n")}\n\n${MEMORY_DRIFT_CAVEAT}\n</memory>`;

  if (text.length > MAX_CONTEXT_CHARS) {
    text = text.slice(0, MAX_CONTEXT_CHARS - 20) + "\n... (truncated)\n</memory>";
  }

  return { text, stats: { semantic: semanticCount, lessons: lessonCount } };
}

// ─── Helpers ─────────────────────────────────────────────────────────

/** Staleness thresholds (in days) */
const STALE_WARNING_DAYS = 30;
const VERY_STALE_DAYS = 90;

function formatSection(title: string, items: string[]): string {
  return `## ${title}\n${items.map(i => `- ${i}`).join("\n")}`;
}

/**
 * Format a semantic entry with staleness indicator.
 * Memories older than 30 days get a warning; older than 90 days get a strong warning.
 * This prevents the agent from treating stale facts as current truth.
 */
function formatSemantic(entry: SemanticEntry): string {
  const key = entry.key.split(".").slice(1).join(".");
  const ageDays = daysSince(entry.updated_at);
  const staleTag = ageDays >= VERY_STALE_DAYS
    ? ` ⚠️ ${ageDays}d old — verify before acting on this`
    : ageDays >= STALE_WARNING_DAYS
      ? ` (${ageDays}d ago)`
      : "";
  return `${key}: ${entry.value}${staleTag}`;
}

/**
 * Calculate days since a date string.
 */
function daysSince(dateStr: string): number {
  try {
    const then = new Date(dateStr).getTime();
    const now = Date.now();
    return Math.floor((now - then) / (1000 * 60 * 60 * 24));
  } catch {
    return 0;
  }
}

/**
 * Memory drift caveat — appended to the memory block so the agent knows
 * to verify recalled facts against current state before acting on them.
 */
const MEMORY_DRIFT_CAVEAT = `## Before acting on memory
- Memory records can become stale. If a memory names a file, function, or flag — verify it still exists before recommending it. "The memory says X exists" is not the same as "X exists now."
- If a recalled memory conflicts with what you observe in the current code or project state, trust what you observe now.
- Memories about project state (deadlines, decisions, architecture) decay fastest — check if still relevant.

## Searching memory proactively
- Use \`memory_search\` to find specific facts relevant to your current task — the injected block above may be truncated or may not cover the angle you need.
- Use \`memory_lessons\` with a category filter to review relevant learned corrections before acting.
- Use \`memory_stats\` for a quick overview of store health.`;

export function projectSlug(cwd: string): string {
  const parts = cwd.split("/").filter(Boolean);
  const skip = new Set(["workplace", "local", "home", "src", "scratch", os.userInfo().username]);
  for (const p of parts.reverse()) {
    if (!skip.has(p.toLowerCase()) && p.length > 1) return p.toLowerCase();
  }
  return "";
}
