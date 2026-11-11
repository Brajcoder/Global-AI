# ChronosBench + VerityGate: Review Fixes and Next Steps

This document records what the code review found, what has been fixed, and the recommended
order of work from here. How to run everything is in `README.md`; the demo script lives in
`IMPLEMENTATION_AND_RUNBOOK.md`.

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
| Tests | 4 trivial tests | An offline suite covering every bug above and all features in section 2. Run with `python -m pytest` |

### Behaviour changes to be aware of

- `usage` in `/verify` responses changed: `tavily_extract_calls` and `cache_hit` were replaced by
  `evidence_items_scored`, `cache_hits`, `search_errors`, `credits_used` and `offline_mode`.
- `overall_decision` can now be `clean`, `stale_detected`, `needs_review` or `no_claims`.
- `/benchmarks/run` responses include `telemetry_source` (`provider_usage`, `estimated` or `mixed`).
- Truth scores are lower and more honest than before. Offline, most claims are `UNVERIFIED`
  because only the Pydantic fixture returns evidence. Use a real Tavily key for meaningful numbers.

---

## 2. What was built after the review

| Step from the original plan | Status | Where |
|---|---|---|
| LLM stance judge with heuristic fallback | Done | `packages/evidence_engine/llm_judge.py` |
| OpenAI-compatible providers (model + key + base URL) | Done | `integrations/llm/`, `POST /api/v1/providers` |
| Real agents driven by the manifest's model | Done | `agents/runtime.py`, `agents/*.py` profiles |
| External agents over HTTP (`endpoint_or_command`) | Done | `agents/runtime.py: run_http_agent` |
| Real telemetry (tokens from the provider, measured latency, loop detection) | Done | `telemetry_source` on every score |
| Background benchmark jobs, progress and a persisted leaderboard | Done | `apps/api/routers/benchmarks.py` |
| API key on write endpoints, configurable CORS, agent ID validation | Done | `apps/api/security.py` |
| Adding new columns to an existing database on startup | Done | `packages/contracts/database.py` |
| Dataset grown from 5 to 15 tasks, plus a focus-coverage metric | Done | `benchmarks/datasets/temporal_traps.json` |
| Dashboard: claim explorer, leaderboard, agents, providers | Done | `apps/web/index.html` |
| CI | Done | `.github/workflows/ci.yml` |

Bifrost needs no special integration: it is an OpenAI-compatible gateway, so it is the `bifrost`
provider preset (`BIFROST_GATEWAY_URL`). Token usage comes from its responses like any other provider.

## 3. Known limitations (still open)

1. **Offline mode is only a demo.** Without `TAVILY_API_KEY` only Pydantic-related claims find
   evidence. Without an LLM the heuristic judge is used; it cannot understand negation, numbers
   ("$5" vs "$2.50") or paraphrase, and flags a meta-statement such as "some posts still mention
   dict()" as STALE. The LLM judge is there to handle these.
2. **Benchmark tasks have no ground-truth answers.** Truth is scored against live evidence, so
   scores move when search results change. `focus_coverage` is a keyword check, not a correctness check.
3. **Background runs live in the server process.** A restart marks unfinished runs `failed`, and
   running several uvicorn workers is not supported. A real job queue would fix both.
4. **No rate limiting.** `API_KEY` protects write endpoints, but a valid key can still spend credits freely.
5. **Provider API keys are stored unencrypted** in the database (never returned by the API).
6. **No real migrations.** New columns are added automatically; renames and type changes are not handled.
7. **The Tavily cache is a local SQLite file**, even when `DATABASE_URL` points at Postgres.
8. **MCP servers and native tools in the manifest are recorded but not executed.** The built-in
   profiles only have `web_search`.

## 4. Recommended next steps

1. **Ground truth per task**: expected current facts and known stale facts with dates, scored
   alongside live evidence. This makes the benchmark reproducible.
2. **Regression snapshots**: fail CI when a reference agent's score drifts by more than N points.
3. **Rate limiting** on `/verify` and `/benchmarks/run` (for example `slowapi`).
4. **Postgres + Alembic** if the host has an ephemeral disk: add `psycopg[binary]`, set
   `DATABASE_URL=postgresql+psycopg://...` (Supabase: use the direct connection or session pooler on
   port 5432), and move the Tavily cache into a SQLModel table.
5. **A job queue** (or at least a worker process) for benchmarks, so runs survive restarts.
6. **Encrypt stored provider keys**, or read them from a secret manager.

## 5. Team conventions

- Run `python -m pytest` before pushing; add a test with every bug fix.
- Never commit `.env` or `data/` (both are git-ignored and docker-ignored).
- Keep scoring constants in `packages/scoring_engine/scoring.py` and document any weight change
  in the runbook, since judges will compare the numbers against the formulas.
