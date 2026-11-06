# ChronosBench + VerityGate: Review Fixes and Next Steps

This document records what the code review found, what has been fixed, and the recommended
order of work from here. Setup and the demo script live in `IMPLEMENTATION_AND_RUNBOOK.md`.

---

## 1. What was fixed

| Area | Problem | Fix |
|---|---|---|
| Persistence | Claim IDs were `clm_001`, `clm_002`… per run and used as the DB primary key, so the **second** `/verify` call returned 500 | Claim IDs are now globally unique (`clm_<uuid>`) |
| Benchmark | Crashed on the 2nd task (same bug); a failure left the run stuck at `running` forever | Agents and suite are validated before the run is created; failures mark the run `failed` |
| Startup | Fresh clone could not start: SQLite cannot create the missing `data/` folder | `data/` is created automatically; SQLite, cache, `.env`, UI and dataset paths resolve against the project root, so the server can start from any directory |
| Claim extraction | Splitting on "and"/"but" turned "use .dict() and @validator" into fragments, so the headline stale claim was classified `stable` and scored 100 | Splits only on sentences and `;`; keeps exact offsets; doesn't split after "e.g." / "i.e." |
| Classifier | Substring matching (`"live"` matched "deliver", `"api"` matched "rapid", `"version"` matched "conversion"); code usage like `.dict()` was never checked | Word-boundary patterns; code-API patterns (`.x()`, `x()`, `@x`) are checked live; hedging words no longer skip checkable claims |
| Stance | `"is"` was a "supports" hint (matched "this", "lists"), so nearly everything "supported" | Term-overlap based support; directional "X replaced by Y" detection; irrelevant evidence is labelled `irrelevant` |
| Hard-coded demo rule | `dict()` / `@validator` were forced to STALE in code | Removed. STALE now comes from evidence ("Y replaces X" + claim uses X) with a concrete patch: *replace `dict` with `model_dump`* |
| Scoring | Stable/opinion claims got 100/100/100 (vague answers won); missing publish date scored as perfectly fresh; irrelevant evidence inflated scores | Unchecked claims get neutral scores (≈77.5 per claim); unknown date = 50; claim scores use only the evidence that decided the status |
| Tavily | Offline fallback returned the Pydantic page for **every** query and was cached under the real query forever; cache never expired; settings were ignored; credits miscounted | Offline fixtures are keyword-matched and never cached; 24h cache TTL; `MAX_CLAIMS_PER_REQUEST` enforced; search depth configurable; credits counted correctly (advanced = 2) |
| Resilience | One failed Tavily call returned 500 for the whole request | Search failures mark only that claim `UNVERIFIED` and are counted in `usage.search_errors` |
| Performance | Tavily searches ran one after another | Searches run concurrently (`asyncio.gather`) |
| API behaviour | Empty text returned `clean`; `strict` mode did nothing | Empty → `no_claims`; mixed results → `needs_review`; strict mode returns `recommended_action: block` for stale answers |
| Benchmark agents | Stub agents echoed the task prompt, which was then scored as the agent's claims | Echo removed |
| Misc | `init_db()` on every request; deprecated `on_event`; naive `utcnow()`; invalid CORS wildcard+credentials; re-registering an agent reset `created_at`; UI crashed silently on errors | All fixed |
| Docker | `.env`, `data/`, `.venv` copied into the image; compose failed without `.env` | Added `.dockerignore`; `.env` is optional in compose |
| Tests | 4 trivial tests | 30 tests, including API tests that would have caught every bug above. Run with `python -m pytest` |

### Behaviour changes to be aware of

- `usage` in `/verify` responses changed: `tavily_extract_calls` and `cache_hit` were replaced by
  `evidence_items_scored`, `cache_hits`, `search_errors`, `credits_used` and `offline_mode`.
- `overall_decision` can now be `clean`, `stale_detected`, `needs_review` or `no_claims`.
- `/benchmarks/run` responses include `telemetry_source: "estimated"`.
- Truth scores are lower and more honest than before. Offline, most claims are `UNVERIFIED`
  because only the Pydantic fixture returns evidence. Use a real Tavily key for meaningful numbers.

---

## 2. Known limitations (still open)

1. **Stance detection is heuristic.** It catches "X is deprecated / replaced by Y" patterns and
   term overlap. It cannot understand negation, numbers ("$5" vs "$2.50") or paraphrase. It also
   still flags a meta-statement such as "some posts still mention dict()" as STALE.
2. **Agents are canned strings.** They ignore the task and the registered model config, so the
   benchmark compares three fixed paragraphs, not three agents.
3. **Efficiency telemetry is estimated** from output word count, so EffScore is a proxy for
   answer length.
4. **Benchmarks run inside one HTTP request.** A full run with live Tavily calls can take
   minutes and hit client or proxy timeouts.
5. **No authentication.** Anyone who can reach the API can overwrite registered agents and
   spend Tavily credits.
6. **No migrations.** `create_all()` creates tables but never alters them.
7. **Only one benchmark suite**, with 5 tasks and no ground-truth answers.

---

## 3. Recommended next steps (in priority order)

### Step 1: LLM stance judge (biggest quality gain) · ~0.5 day

Replace `infer_stance` with an LLM call and keep the heuristic as the offline fallback.

- Add `packages/evidence_engine/llm_judge.py` with an async `judge(claim, excerpt, published_at)`
  that returns `{stance, confidence, outdated_term, replacement, rationale}` as JSON.
- Call it through the Bifrost gateway (OpenAI-compatible) or directly via Groq/Gemini using
  `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL`.
- Prompt rules: judge only from the excerpt; treat the excerpt as untrusted data (prompt-injection
  defence); return `irrelevant` when the excerpt does not address the claim.
- Use the judge's `replacement` to write the `suggested_patch`.
- Cache judgements by `(claim, url)` hash so repeated demos cost nothing.
- Keep `stancer.py` as the fallback when no LLM key is configured, so tests stay offline.

### Step 2: Real agents · ~1 day

- Define an agent interface `async run_task(prompt, manifest) -> AgentRun` that returns
  `output`, `tool_calls`, `steps`, `prompt_tokens`, `completion_tokens` and `latency_ms`.
- Implement the three reference profiles with real LLM calls that use the manifest's model,
  temperature and system prompt:
  - **parametric_baseline**: a single LLM call with no tools.
  - **unbounded_deep_research**: Tavily search in a loop with a high step limit.
  - **disciplined_verifier**: search, then verify, capped at `MAX_AGENT_STEPS_PER_TASK`.
- Replace the hard-coded `AGENT_MODULES` map by using `endpoint_or_command` from the manifest:
  an HTTP endpoint to POST the task to, or a Python module path. This is what makes the
  platform "framework-agnostic".

### Step 3: Real telemetry through Bifrost · ~0.5 day

- Route every agent LLM call through `BIFROST_GATEWAY_URL` and read token usage and latency
  from the gateway responses or logs, instead of `estimate_telemetry`.
- Record each tool call in `invoked_tool_calls_json`. Detect loops as repeated identical tool
  calls or queries; this gives `loop_count` a real meaning.
- Set `telemetry_source: "bifrost"` once wired.

### Step 4: Background benchmark jobs · ~0.5 day

- `POST /benchmarks/run` returns `202` and a `benchmark_id` immediately; run the work in a
  FastAPI `BackgroundTasks` or `asyncio.create_task`.
- `GET /benchmarks/{id}` reports progress (`tasks_done / tasks_total`) and the leaderboard
  once it's complete.
- Persist the leaderboard (a new `benchmark_scores` table) so finished runs can be re-displayed.

### Step 5: Database decision (SQLite vs Supabase)

| Situation | Choice |
|---|---|
| Demo runs on a laptop or a VM with a persistent disk | **Stay on SQLite.** Nothing to do. |
| Deploying to Render, Railway or any host with an ephemeral disk | **Move to Supabase (Postgres).** Otherwise data and the Tavily cache vanish on every restart |
| Teammates need to share data or you want a browsable table UI | **Supabase** |

How to switch (the code is already database-agnostic through SQLModel):

1. Add `psycopg[binary]` to `requirements.txt`.
2. Set `DATABASE_URL=postgresql+psycopg://...` using Supabase's **direct connection or session
   pooler** (port 5432). The transaction pooler (6543) conflicts with prepared statements.
3. Move the Tavily cache from raw `sqlite3` into a SQLModel table so it persists too.
4. Add Alembic (`alembic init`, autogenerate from `SQLModel.metadata`) before the first schema
   change. `create_all()` will not alter existing tables.

You don't need the `supabase-py` SDK; treat Supabase as plain Postgres.

### Step 6: Security basics before any public deployment · ~0.5 day

- An API key header (`X-API-Key`) checked by a FastAPI dependency on all write endpoints.
- Restrict CORS `allow_origins` to the dashboard's domain.
- Rate-limit `/verify` and `/benchmarks/run`, for example with `slowapi`, to protect Tavily credits.
- Validate `agent_id` format (e.g. `^[a-z0-9_]{3,64}$`).

### Step 7: Dataset and evaluation · ongoing

- Grow `temporal_traps.json` to 30+ tasks across the cohorts, and add **ground truth** per task:
  the expected current facts and known stale facts with dates.
- Score agents against ground truth as well as live evidence; this makes the benchmark
  reproducible when Tavily results change.
- Add benchmark regression snapshots: fail CI if a reference agent's score drifts by more than
  N points.

### Step 8: Dashboard · 1–2 days

The current UI is a single verify box. For the demo, at minimum add:

- Leaderboard table (TruthScore / EffScore / ChronosScore).
- Claim explorer: text with highlighted claims coloured by status, evidence and patch on click.
- Claim history timeline for reverify demonstrations.

A static page that calls the existing endpoints is enough; Next.js is optional.

### Step 9: CI · ~1 hour

A GitHub Actions workflow that installs `requirements.txt` and runs `python -m pytest` on every
push. The tests are offline and need no secrets.

---

## 4. Hackathon-critical path

If time is short, do these and skip the rest:

1. **Step 1** (LLM judge): makes stale detection credible beyond the Pydantic example.
2. **Step 2** (real agents, even minimal): makes the leaderboard meaningful.
3. **Step 4** (background jobs): prevents timeouts during the live demo.
4. Decide on hosting, then **Step 5** only if the host's disk is ephemeral.
5. Record the demo with a real Tavily key, and pre-warm the cache by running the demo once.

## 5. Team conventions

- Run `python -m pytest` before pushing; add a test with every bug fix.
- Never commit `.env` or `data/` (both are git-ignored and docker-ignored).
- Keep scoring constants in `packages/scoring_engine/scoring.py` and document any weight change
  in the runbook, since judges will compare the numbers against the formulas.
