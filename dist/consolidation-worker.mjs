#!/usr/bin/env node
/**
 * Background consolidation worker for pi-memory.
 *
 * Spawned as a detached child process during pi shutdown so the user gets
 * their prompt back immediately while memory extraction completes in the
 * background.
 *
 * Usage: node consolidation-worker.mjs <data-file>
 *
 * The data file is a JSON object written by the main pi process containing:
 *   userMessages, assistantMessages, sessionCwd, sessionId,
 *   dbPath, model, facts (existing), lessons (existing)
 */
import { DatabaseSync } from "node:sqlite";
import { readFileSync, unlinkSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import { spawnSync } from "node:child_process";

// ─── Store (lightweight, just what we need) ────────────────────────

class WorkerStore {
  constructor(dbPath) {
    const dir = dirname(dbPath);
    if (!existsSync(dir)) {
      console.error("pi-memory worker: DB directory does not exist", dir);
      process.exit(1);
    }
    this.db = new DatabaseSync(dbPath);
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA busy_timeout = 5000");
    this.db.exec("PRAGMA foreign_keys = ON");
    // Schema should already exist, but be safe
    this.ensureSchema();
  }

  ensureSchema() {
    try {
      this.db.exec(`SELECT 1 FROM semantic LIMIT 1`);
    } catch {
      // Tables don't exist — main process should have created them
      console.error("pi-memory worker: semantic table not found, skipping");
      process.exit(1);
    }
  }

  listSemantic(prefix, limit = 200) {
    if (prefix) {
      return this.db.prepare("SELECT * FROM semantic WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?").all(`${prefix}%`, limit);
    }
    return this.db.prepare("SELECT * FROM semantic ORDER BY updated_at DESC LIMIT ?").all(limit);
  }

  listLessons(prefix, limit = 100) {
    return this.db.prepare("SELECT * FROM lessons WHERE is_deleted = 0 ORDER BY created_at DESC LIMIT ?").all(limit);
  }

  setSemantic(key, value, confidence = 0.8, source = "consolidation") {
    const normalized = key.toLowerCase();
    this.withLock(() => {
      const existing = this.db.prepare("SELECT * FROM semantic WHERE key = ?").get(normalized);
      if (existing && existing.confidence > confidence) return;
      this.db.prepare(`
        INSERT INTO semantic (key, value, confidence, source, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
          value = excluded.value,
          confidence = excluded.confidence,
          source = excluded.source,
          updated_at = datetime('now')
      `).run(normalized, value, confidence, source);
      this.logEvent(existing ? "update" : "create", "semantic", normalized);
    });
  }

  addLesson(rule, category = "general", source = "consolidation", negative = false, project) {
    const trimmed = rule.trim();
    if (!trimmed) return { success: false, reason: "empty rule" };
    const normalizedCategory = category.trim().toLowerCase() || "general";
    return this.withLock(() => {
      const existing = this.db.prepare(
        "SELECT id FROM lessons WHERE LOWER(TRIM(rule)) = LOWER(?) AND is_deleted = 0"
      ).get(trimmed.toLowerCase());
      if (existing) return { success: false, reason: "duplicate", id: existing.id };
      const allRules = this.db.prepare("SELECT id, rule FROM lessons WHERE is_deleted = 0").all();
      for (const r of allRules) {
        if (jaccard(trimmed, r.rule) >= 0.7) {
          return { success: false, reason: "similar", id: r.id };
        }
      }
      const id = crypto.randomUUID();
      this.db.prepare(
        "INSERT INTO lessons (id, rule, category, source, negative, project) VALUES (?, ?, ?, ?, ?, ?)"
      ).run(id, trimmed, normalizedCategory, source, negative ? 1 : 0, project ?? null);
      this.logEvent("create", "lesson", id, trimmed.slice(0, 100));
      return { success: true, id };
    });
  }

  withLock(fn) {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const result = fn();
      this.db.exec("COMMIT");
      return result;
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }
  }

  logEvent(eventType, memoryType, key, details = "") {
    this.db.prepare(
      "INSERT INTO events (event_type, memory_type, memory_key, details) VALUES (?, ?, ?, ?)"
    ).run(eventType, memoryType, key, details);
  }

  close() {
    this.db.close();
  }
}

// ─── Consolidation logic (mirrors pi-memory's consolidator) ────────

const CONSOLIDATION_PROMPT = `You are a memory extraction system. Analyze this conversation and extract structured knowledge.

Extract ONLY concrete, reusable facts — not summaries of what happened. Focus on:

1. **User preferences** (key prefix: pref.) — coding style, tool preferences, workflow habits
   Example: { "key": "pref.commit_style", "value": "conventional commits", "confidence": 0.9 }

2. **Project patterns** (key prefix: project.<name>.) — languages, frameworks, architecture decisions
   Example: { "key": "project.rosie.di", "value": "Dagger dependency injection", "confidence": 0.95 }

3. **Tool preferences** (key prefix: tool.) — which tools to prefer/avoid, how to use them
   Example: { "key": "tool.sed", "value": "use for daily note insertion, not echo >>", "confidence": 0.9 }

4. **Corrections/lessons** — things the user corrected, mistakes to avoid
   Example: { "rule": "Use sed to insert after ## Notes heading, not echo >> which appends after Tags", "category": "vault", "negative": true }

5. **Validated approaches** — things the user explicitly confirmed worked well (positive signal)
   Example: { "rule": "When deploying wiki changes, draft first and let user preview before publishing", "category": "wiki-edit", "negative": false }

## What NOT to extract — these are derivable or ephemeral, and pollute memory:

- **Code patterns, architecture, file paths, project structure** — these can be derived by reading the current project state (grep, git, file reads)
- **Git history, recent changes, who-changed-what** — git log/blame are authoritative
- **Debugging solutions or fix recipes** — the fix is in the code; the commit message has context
- **Anything already documented in AGENTS.md, CLAUDE.md, or project config files**
- **Ephemeral task details** — in-progress work, temporary state, current conversation context
- **Activity summaries** — "today we worked on X" is not a lasting fact. Instead ask: what was *surprising* or *non-obvious* about it?
- **File contents or code snippets** — the file itself is the source of truth
- **Exact commands that worked once** — unless they encode a non-obvious pattern that the agent consistently gets wrong

These exclusions apply even if the user asks to save such things. If asked, extract what was *surprising* or *non-obvious* — that is the part worth keeping.

Rules:
- Only extract if confidence >= 0.8 (you're reasonably sure this is a lasting preference, not a one-off)
- Key format: lowercase, dots as separators, no spaces
- Keep values concise (under 200 chars)
- For corrections, set negative=true if it's something to AVOID
- For validated approaches (user confirmed something works), set negative=false

Respond with ONLY valid JSON matching this schema:
{
  "semantic": [{ "key": "string", "value": "string", "confidence": number }],
  "lessons": [{ "rule": "string", "category": "string", "negative": boolean }]
}

If nothing worth extracting, return: { "semantic": [], "lessons": [] }`;

function buildConsolidationPrompt(input, currentFacts, currentLessons) {
  const messages = [];
  let memorySection = "";
  if ((currentFacts && currentFacts.length > 0) || (currentLessons && currentLessons.length > 0)) {
    const parts = ["## Current Memory State"];
    if (currentFacts && currentFacts.length > 0) {
      parts.push("The user already has these facts stored (avoid duplicating, update if changed):");
      let chars = 0;
      for (const f of currentFacts) {
        const line = `- ${f.key}: ${f.value.length > 120 ? f.value.slice(0, 120) + "…" : f.value}`;
        if (chars + line.length > 1500) { parts.push("- ... (truncated)"); break; }
        parts.push(line);
        chars += line.length;
      }
    }
    if (currentLessons && currentLessons.length > 0) {
      parts.push("\nAnd these lessons (avoid duplicating):");
      let chars = 0;
      for (const l of currentLessons) {
        const line = `- [${l.category}] ${l.rule.length > 120 ? l.rule.slice(0, 120) + "…" : l.rule}`;
        if (chars + line.length > 500) { parts.push("- ... (truncated)"); break; }
        parts.push(line);
        chars += line.length;
      }
    }
    memorySection = parts.join("\n") + "\n\n";
  }
  const maxPairs = 30;
  const len = Math.min(input.userMessages.length, maxPairs);
  for (let i = 0; i < len; i++) {
    const userMsg = input.userMessages[i];
    if (userMsg) messages.push(`User: ${truncate(userMsg, 1000)}`);
    const assistantMsg = input.assistantMessages[i];
    if (assistantMsg) messages.push(`Assistant: ${truncate(assistantMsg, 500)}`);
  }
  return `${CONSOLIDATION_PROMPT}\n\n${memorySection}${input.cwd ? `Working directory: ${input.cwd}\n` : ""}\n## Conversation\n\n${messages.join("\n\n")}`;
}

const VALID_KEY_RE = /^[a-z][a-z0-9._-]*$/;

function isValidKey(key) {
  return VALID_KEY_RE.test(key) && key.length <= 100 && key.length >= 2;
}

function isDerivableOrEphemeral(key, value) {
  const kl = key.toLowerCase();
  const vl = value.toLowerCase();
  if (kl.includes("filepath") || kl.includes("file_path") || kl.includes("directory")) return true;
  if (/^project\.\w+\.(path|dir|location|structure|layout|architecture)$/.test(kl)) return true;
  if (kl.includes("commit") || kl.includes("git.history") || kl.includes("git.recent")) return true;
  if (vl.startsWith("today ") || vl.startsWith("we worked on") || vl.startsWith("this session")) return true;
  if (vl.includes("```") && vl.length > 300) return true;
  if (kl.includes("current_task") || kl.includes("in_progress") || kl.includes("investigating")) return true;
  return false;
}

function isDerivableLesson(rule) {
  const rl = rule.toLowerCase();
  if (/file .+ is (at|in|located) /.test(rl)) return true;
  if (/^the (project|codebase|repo) (uses|is written in) /.test(rl)) return true;
  if (/^(we|i|the agent) (fixed|deployed|updated|changed|modified|ran|executed) /.test(rl)) return true;
  if (/^when (encountering|bash fails|edit fails|.*error)/.test(rl) && /\b(run:|fix with:)/.test(rl)) return true;
  if (/^run: /.test(rl)) return true;
  if (rl.includes("command exited with code") && rl.length < 100) return true;
  return false;
}

function truncate(text, max) {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

function parseConsolidationResponse(text) {
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || text.match(/(\{[\s\S]*\})/);
  if (!jsonMatch) return { semantic: [], lessons: [] };
  try {
    const parsed = JSON.parse(jsonMatch[1].trim());
    const result = { semantic: [], lessons: [] };
    if (Array.isArray(parsed.semantic)) {
      for (const s of parsed.semantic) {
        if (typeof s.key === "string" && typeof s.value === "string" && typeof s.confidence === "number") {
          if (s.confidence >= 0.8 && isValidKey(s.key) && s.value.length <= 500) {
            result.semantic.push({ key: s.key, value: s.value, confidence: s.confidence });
          }
        }
      }
    }
    if (Array.isArray(parsed.lessons)) {
      for (const l of parsed.lessons) {
        if (typeof l.rule === "string" && l.rule.trim().length > 0) {
          result.lessons.push({
            rule: l.rule.trim(),
            category: typeof l.category === "string" ? l.category : "general",
            negative: !!l.negative,
          });
        }
      }
    }
    return result;
  } catch {
    return { semantic: [], lessons: [] };
  }
}

function applyExtracted(store, extracted, source = "consolidation", project) {
  let semanticCount = 0;
  let lessonCount = 0;
  for (const s of extracted.semantic) {
    if (isDerivableOrEphemeral(s.key, s.value)) continue;
    store.setSemantic(s.key, s.value, s.confidence, "consolidation");
    semanticCount++;
  }
  for (const l of extracted.lessons) {
    if (isDerivableLesson(l.rule)) continue;
    const lessonProject = source === "user" ? undefined : project;
    const result = store.addLesson(l.rule, l.category, source, l.negative, lessonProject);
    if (result.success) lessonCount++;
  }
  return { semantic: semanticCount, lessons: lessonCount };
}

function jaccard(a, b) {
  const setA = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const setB = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  if (setA.size === 0 && setB.size === 0) return 1;
  const intersection = new Set([...setA].filter((x) => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}

function projectSlug(cwd) {
  const parts = cwd.split("/").filter(Boolean);
  const skip = new Set(["workplace", "local", "home", "src", "scratch"]);
  for (const p of parts.reverse()) {
    if (!skip.has(p.toLowerCase()) && p.length > 1) return p.toLowerCase();
  }
  return "";
}

// ─── Main ──────────────────────────────────────────────────────────

function main() {
  const dataFile = process.argv[2];
  if (!dataFile) {
    console.error("pi-memory worker: missing data file argument");
    process.exit(1);
  }

  let data;
  try {
    data = JSON.parse(readFileSync(dataFile, "utf-8"));
  } catch (err) {
    console.error("pi-memory worker: failed to read data file:", err.message);
    process.exit(1);
  }

  const { userMessages, assistantMessages, sessionCwd, sessionId, dbPath, model, facts, lessons } = data;

  if (!dbPath) {
    console.error("pi-memory worker: no dbPath in data file");
    process.exit(1);
  }

  const store = new WorkerStore(dbPath);

  try {
    // Build consolidation input
    const input = {
      userMessages: userMessages || [],
      assistantMessages: assistantMessages || [],
      cwd: sessionCwd || "",
      sessionId: sessionId || "",
    };

    const currentFacts = facts || store.listSemantic(undefined, 200).map((f) => ({ key: f.key, value: f.value }));
    const currentLessons = lessons || store.listLessons(undefined, 100).map((l) => ({ rule: l.rule, category: l.category }));
    const prompt = buildConsolidationPrompt(input, currentFacts, currentLessons);

    // Call pi for LLM extraction with a reasonable timeout
    const piModel = model || "opencode-go/deepseek-v4-flash";
    const TIMEOUT_MS = 90_000; // 90s — generous since user is already back at shell

    const result = spawnSync("pi", [
      "-p", prompt,
      "--print",
      "--no-extensions",
      "--no-tools",
      "--no-session",
      "--model", piModel,
    ], {
      timeout: TIMEOUT_MS,
      cwd: sessionCwd || process.cwd(),
      stdio: ["ignore", "pipe", "pipe"],
      maxBuffer: 10 * 1024 * 1024, // 10MB
    });

    if (result.status === 0 && result.stdout) {
      const stdout = result.stdout.toString();
      const extracted = parseConsolidationResponse(stdout);
      const slug = sessionCwd ? projectSlug(sessionCwd) : "";
      const applied = applyExtracted(store, extracted, `session:${sessionId || "unknown"}`, slug || undefined);
      if (applied.semantic + applied.lessons > 0) {
        console.error(`pi-memory worker: consolidated ${applied.semantic} facts, ${applied.lessons} lessons`);
      }
    } else if (result.error) {
      console.error("pi-memory worker: pi exec failed:", result.error.message);
    } else {
      const stderr = result.stderr?.toString().slice(0, 500) || "";
      console.error(`pi-memory worker: pi exited code=${result.status}, stderr:`, stderr);
    }
  } catch (err) {
    console.error("pi-memory worker: consolidation failed:", err.message);
  } finally {
    store.close();
    // Clean up the data file
    try { unlinkSync(dataFile); } catch { /* best effort */ }
  }
}

main();
