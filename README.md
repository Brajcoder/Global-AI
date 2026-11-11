# ChronosBench + VerityGate AI

Two tools in one service:

- **VerityGate** checks an answer for stale facts. It splits the text into claims, searches the live
  web (Tavily) for the time-sensitive ones, judges each claim against the evidence and suggests a fix:
  *"replace `dict` with `model_dump`"*.
- **ChronosBench** benchmarks research agents on "temporal trap" questions and ranks them on truth
  **and** efficiency: `ChronosScore = 0.65 x TruthScore + 0.35 x EffScore`.

It works with **any OpenAI-compatible model provider**: give it a base URL, an API key and a model name.
With no keys at all it still runs end to end in offline mode, so you can try it immediately.

## Quick start

Requires Python 3.11 or newer.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn apps.api.main:app --port 8000
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m uvicorn apps.api.main:app --port 8000
```

Then open:

- **http://localhost:8000**: the dashboard (verify, benchmark, agents, model providers)
- **http://localhost:8000/docs**: interactive API reference

The coloured pills at the top of the dashboard show what the server is using: live search or offline
fixtures, and which default LLM (if any).

### Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

`.env` is read if it exists, and the database is kept in `./data`.

## Configure it

Everything is set in `.env` (see `.env.example` for the full list). All of it is optional.

| What you set | What you get |
|---|---|
| nothing | Offline mode: a built-in Pydantic fixture as evidence, heuristic judge, canned agent answers |
| `TAVILY_API_KEY` | Real live web evidence for claims and agent searches |
| an LLM (below) | Real agents making real model calls, and an LLM stance judge |
| `API_KEY` | Write endpoints require the header `X-API-Key` |

### Model providers (OpenAI-compatible)

Anything that serves `POST {base_url}/chat/completions` works: OpenAI, Groq, Gemini, OpenRouter,
Together, DeepSeek, Mistral, Ollama, LM Studio, vLLM, LiteLLM, Bifrost. There are four ways to point
the app at a model. Use whichever fits.

**1. A default for the whole server**, in `.env`:

```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

**2. A built-in preset**, in `.env`. Only the key is needed because the base URL is known:

```ini
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```

Presets: `openai`, `groq`, `gemini`, `openrouter`, `together`, `deepseek`, `mistral` (each reads
`<NAME>_API_KEY`), plus `ollama`, `lmstudio` and `bifrost`, which need no key.

**3. A named provider registered at runtime**, from the dashboard's *Model providers* tab or the API:

```bash
curl -X POST http://localhost:8000/api/v1/providers -H "Content-Type: application/json" -d '{
  "name": "my-provider",
  "base_url": "https://api.example.com/v1",
  "api_key": "sk-...",
  "default_model": "some-model"
}'

# Check that the URL, key and model actually work:
curl -X POST http://localhost:8000/api/v1/providers/my-provider/test -H "Content-Type: application/json" -d '{}'
```

**4. Inline, per agent or per request**: pass `base_url`, `api_key` and the model name directly.
See the examples below.

How a provider name is resolved: inline `base_url` first, then a registered provider, then a preset
with a key in `.env`. If none of those match, the server default connection **and its model** are used.
The provider name `default` uses the server default connection with the model name you give.

API keys are stored in the local database so the server can use them. They are never returned by the
API (responses show a masked form such as `sk-...cdef`). Set `API_KEY` before exposing the server.

## Use it

### Verify an answer

```bash
curl -X POST http://localhost:8000/api/v1/verify -H "Content-Type: application/json" -d '{
  "text": "To serialize in Pydantic, use .dict() and @validator.",
  "mode": "strict"
}'
```

The response lists each claim with a status (`VERIFIED`, `STALE`, `CONTRADICTED`, `UNVERIFIED`,
`STABLE`, `NEEDS_REVIEW`), the evidence, a `suggested_patch`, the overall `truth_score` and a
`recommended_action` (`return`, `return_with_warning`, or `block` in strict mode).

Choose the judge model for a single request:

```json
{
  "text": "...",
  "options": {"llm": {"provider": "my-provider", "model": "some-model"}}
}
```

or spell out the connection:

```json
{
  "text": "...",
  "options": {"llm": {"base_url": "https://api.example.com/v1", "api_key": "sk-...", "model": "some-model"}}
}
```

`"options": {"llm_judge": false}` forces the heuristic judge. `usage.stance_judge` in the response
says which judge was used (`llm`, `heuristic` or `mixed`).

### Register agents and run a benchmark

An agent is a **behaviour profile** plus a **model**:

| Profile | Behaviour |
|---|---|
| `parametric_baseline` | One model call, no tools. Answers from memory. |
| `disciplined_verifier` | Web search with a budget of two searches, then answers. |
| `unbounded_deep_research` | Searches repeatedly, up to `MAX_AGENT_STEPS_PER_TASK` steps. |

```bash
# Uses a named provider
curl -X POST http://localhost:8000/api/v1/agents/register -H "Content-Type: application/json" -d '{
  "agent_id": "verifier_small",
  "name": "Verifier on a small model",
  "profile": "disciplined_verifier",
  "model_config": {"provider": "my-provider", "model_name": "some-model", "temperature": 0.1}
}'

# Or give the connection inline
curl -X POST http://localhost:8000/api/v1/agents/register -H "Content-Type: application/json" -d '{
  "agent_id": "baseline_local",
  "name": "Baseline on Ollama",
  "profile": "parametric_baseline",
  "model_config": {"model_name": "llama3.1", "base_url": "http://localhost:11434/v1"}
}'

# Run in the background and poll
curl -X POST http://localhost:8000/api/v1/benchmarks/run -H "Content-Type: application/json" -d '{
  "agent_ids": ["verifier_small", "baseline_local"],
  "limit_tasks": 3,
  "background": true
}'
curl http://localhost:8000/api/v1/benchmarks/<benchmark_id>
```

Or do the same from the dashboard, or from the command line:

```bash
python -m benchmarks.evaluation.evaluate --run --agents verifier_small baseline_local --limit 3
```

Each agent in the list shows how it will run: `llm` (real model calls), `offline` (no model resolved,
canned answer) or `external`.

**Bring your own agent (any framework).** Set `endpoint_or_command` to an http(s) URL. For each task
the server POSTs `{"task_id", "prompt", "agent_id", "model"}` and expects JSON back:

```json
{"output": "the answer text", "usage": {"prompt_tokens": 120, "completion_tokens": 80}, "steps": 3}
```

Only `output` is required.

### How scores are computed

- **TruthScore**: every claim in the agent's answer is verified. Per claim:
  `status multiplier x (0.60 x support + 0.25 x freshness + 0.15 x source quality)`, averaged.
  Freshness halves every 90 days for software and pricing claims (1 day for live status).
- **EffScore**: `100 x (0.50 x e^(-tokens/4000) + 0.30 x e^(-latency/10s) + 0.20 x (1 - loop ratio))`.
  A loop is a repeated identical search.
- **ChronosScore**: `0.65 x TruthScore + 0.35 x EffScore`.
- **Focus %**: share of the task's expected terms the answer mentions. Shown for context only; it is
  not part of ChronosScore.

`telemetry_source` tells you where token counts came from: `provider_usage` (reported by the model
API) or `estimated` (offline agents, or a provider that omits usage).

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Status, search mode, default LLM, whether a key is required |
| POST | `/api/v1/verify` | Verify a text |
| GET | `/api/v1/verifications`, `/api/v1/verification/{id}` | Past verifications with claims and evidence |
| POST | `/api/v1/claims/{id}/reverify` | Re-check one claim; changes are logged |
| GET | `/api/v1/claims/{id}/history` | Status changes of a claim |
| POST / GET / DELETE | `/api/v1/providers`, `/api/v1/providers/{name}` | Manage model providers |
| POST | `/api/v1/providers/{name}/test` | Send a test request to a provider |
| POST / GET / DELETE | `/api/v1/agents/register`, `/api/v1/agents`, `/api/v1/agents/{id}` | Manage agents |
| POST | `/api/v1/benchmarks/run` | Run a benchmark (`"background": true` returns 202) |
| GET | `/api/v1/benchmarks`, `/api/v1/benchmarks/{id}`, `/api/v1/benchmarks/suites` | Runs, progress, leaderboard, suites |

When `API_KEY` is set, every POST and DELETE needs `-H "X-API-Key: <value>"`. GET endpoints stay open.

## Tests

```bash
python -m pytest
```

The suite is fully offline: Tavily runs on fixtures and model calls go to an in-process fake
OpenAI-compatible endpoint, so it needs no keys and spends no credits. CI runs it on every push
(`.github/workflows/ci.yml`).

## Project layout

```
apps/api/            FastAPI app: routers, verification service, API-key guard
apps/web/            Single-file dashboard served at /
agents/              Reference agent profiles and the agent runtime (tool loop, external agents)
integrations/llm/    OpenAI-compatible client and provider resolution
integrations/tavily/ Tavily search client with a TTL cache and offline fixtures
packages/            Claim extraction, volatility classifier, stance judge, scoring, schemas, DB models
benchmarks/          Task dataset and the command-line runner
infra/               Dockerfile and docker-compose
tests/               Offline test suite
```

## Good to know

- **Costs.** With real keys, a benchmark makes real model calls and Tavily searches (1 credit per
  basic search, 2 per advanced; results are cached for 24 hours). Start with `limit_tasks` of 2 or 3.
  `unbounded_deep_research` is deliberately expensive.
- **Offline mode is a demo, not a measurement.** Only Pydantic-related claims find evidence offline,
  and the heuristic judge only understands "X is deprecated / replaced by Y" and term overlap.
- **Tool calling.** Search agents use OpenAI-style tool calls. If a model rejects them, the agent falls
  back to one search on the task text followed by a single answer.
- **Run one server process.** Background benchmarks live in the server process; a restart marks
  unfinished runs as `failed`. Do not run several uvicorn workers.
- **Database.** SQLite in `./data` by default. New columns are added automatically on startup. For
  Postgres, set `DATABASE_URL=postgresql+psycopg://...` and add `psycopg[binary]` to `requirements.txt`.
- **Security.** Registered base URLs and external agent URLs are called by the server, so set `API_KEY`
  and restrict `CORS_ALLOW_ORIGINS` before putting it on a network you do not control. There is no
  rate limiting.

More background: `IMPLEMENTATION_AND_RUNBOOK.md` (demo script, hackathon checklist) and `NEXT_STEPS.md`
(what was built and what is still open).
