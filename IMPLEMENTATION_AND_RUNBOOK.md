# ChronosBench + VerityGate AI

## 1. Project Understanding From Your Hackathon PRD

This project combines two capabilities:

1. ChronosBench: a framework-agnostic benchmarking platform for technical research agents.
2. VerityGate AI: a stale-answer detector that decomposes claims and verifies volatile claims using live sources (Tavily), then scores freshness, support, and source quality.

The key evaluation principle is: truth quality and efficiency are measured separately, then combined.

Scoring equations implemented in this repository:

- Freshness: $F_i = 100 \times 2^{-(age_i / h_c)}$
- TruthScore: $\text{TruthScore} = \frac{1}{N} \sum_i M_i \times (0.60C_i + 0.25F_i + 0.15Q_i)$
- Efficiency: $\text{EffScore} = 100 \times (0.50e^{-T/T_b} + 0.30e^{-L/L_b} + 0.20(1-R_{loop}))$
- Composite: $\text{ChronosScore} = 0.65 \times \text{TruthScore} + 0.35 \times \text{EffScore}$

> **How to run and configure everything (model providers, agents, API reference) is in `README.md`.**
> This file keeps the project background, the demo script and the submission checklist.

## 2. What Is Already Implemented (Code Level)

Implemented end to end, aligned with your document:

1. FastAPI service with required endpoints:
   - GET /api/v1/health
   - POST /api/v1/agents/register
   - GET /api/v1/agents
   - GET /api/v1/agents/{agent_id}
   - POST /api/v1/verify
   - GET /api/v1/verification/{id}
   - GET /api/v1/claims/{id}/history
   - POST /api/v1/claims/{id}/reverify
   - POST /api/v1/benchmarks/run (optionally in the background)
   - GET /api/v1/benchmarks, /api/v1/benchmarks/{id}, /api/v1/benchmarks/suites
   - POST/GET/DELETE /api/v1/providers, POST /api/v1/providers/{name}/test
   - GET /api/v1/verifications, DELETE /api/v1/agents/{id}

2. Data model and persistence (SQLModel + SQLite by default):
   - agent_manifests
   - benchmark_runs
   - agent_trajectories
   - verification_runs
   - claims
   - evidence_items
   - verification_scores
   - claim_history
   - provider_usage

3. Claim pipeline:
   - atomic claim extraction
   - volatility classification
   - live evidence retrieval via Tavily connector (+ SQLite cache)
   - stance judging by an LLM (any OpenAI-compatible provider), with a heuristic fallback
   - per-claim and run-level scoring

4. Benchmarking flow:
   - dataset-driven tasks from temporal_traps.json (15 tasks)
   - 3 reference agent profiles (baseline/deep-research/disciplined) that call the model set in the agent manifest
   - external agents in any framework through an HTTP endpoint
   - real telemetry: provider-reported tokens, measured latency, repeated-search loop detection
     (estimated only when no model is configured)
   - persisted leaderboard with TruthScore, EffScore, ChronosScore

5. Dashboard:
   - verify with highlighted claims, evidence and suggested fixes
   - benchmark runs with live progress and a leaderboard
   - agent and model-provider management

6. Tests (`python -m pytest`):
   - unit tests: scoring, temporal classifier, claim extractor, stance detection, LLM client, LLM judge
   - API tests: verify, strict mode, reverify/history, providers, LLM-driven and external agents,
     background benchmarks, API-key protection
   - tests run offline against a temporary database (see tests/conftest.py)

## 3. Repository Layout You Now Have

- apps/api: FastAPI application
- apps/web: lightweight frontend starter
- agents: reference agent profiles and the agent runtime
- integrations/llm: OpenAI-compatible client and provider resolution
- integrations/tavily: Tavily client + normalizer + cache
- integrations/bifrost_gateway: telemetry estimator used for offline agents
- packages: claim, temporal, evidence, scoring, contracts
- benchmarks/datasets: temporal trap prompts
- tests: unit tests
- infra: docker compose and API Dockerfile

## 4. Where To Use Tavily Credits (Your 9000 Credits)

Use Tavily only where it gives scoring value. Priority order:

1. Claim verification in POST /api/v1/verify:
   - every time-sensitive or highly-volatile claim should call Tavily search/extract.
2. Benchmark runs in POST /api/v1/benchmarks/run:
   - each generated claim from each agent-task output can trigger Tavily calls.
3. Re-verification flow POST /api/v1/claims/{id}/reverify:
   - use for demonstrations of status drift over time.

Credit strategy (important for hackathon demo stability):

1. Keep MAX_CLAIMS_PER_REQUEST at 8 to 10 during demo.
2. Keep MAX_SEARCHES_PER_CLAIM at 2 or 3.
3. Enable Tavily cache (already enabled by default) so repeated demos do not burn credits.
4. Run benchmark with limit_tasks=3 first, then the full 15 tasks for final screenshots.

## 5. What To Do Outside VS Code

Do these tasks in external portals/services:

1. Tavily account:
   - create/generate TAVILY_API_KEY
   - confirm credit balance and rate limits

2. Model providers:
   - get a key for any OpenAI-compatible provider and set it in .env or register it in the
     dashboard (see "Model providers" in README.md)
   - set usage limits so demo cannot exceed budget

3. Optional Bifrost deployment:
   - run Bifrost gateway separately
   - configure route policies and provider keys there
   - set BIFROST_GATEWAY_URL and use the provider name "bifrost"

4. Hackathon submission assets:
   - record short demo video
   - prepare architecture diagram and score screenshots
   - include explanation of stale claim detection with before/after correction examples

## 6. Step-by-Step Run Procedure

### 6.1 Local Run (Fastest)

1. Open a terminal in the repository root (the folder containing `requirements.txt`).
2. Create a virtual environment and install dependencies (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

3. Create env file:

```powershell
Copy-Item .env.example .env
```

4. Edit .env and set at least:
   - TAVILY_API_KEY=your_key
   - LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (or a preset key such as GROQ_API_KEY)
   - Without a Tavily key the API runs in offline mode: only a built-in Pydantic fixture returns evidence
     and every response reports `"offline_mode": true` under `usage`.
   - Without an LLM the agents return canned answers and the stance judge is heuristic.

5. Start API:

```powershell
python -m uvicorn apps.api.main:app --reload --port 8000
```

6. Open:
   - http://localhost:8000
   - http://localhost:8000/docs

### 6.2 Sanity API Calls

1. Health:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

2. Register an agent:

```powershell
$body = @{
  agent_id = "disciplined_verifier"
  name = "Disciplined Verifier"
  framework = "custom"
  model_config = @{
    provider = "groq"
    model_name = "llama-3.3-70b-versatile"
    temperature = 0.1
    top_p = 0.95
    max_tokens = 2048
  }
  prompts = @{
    system_prompt = "Always verify volatile facts"
    instruction_template = "Task: {{prompt}}"
  }
  capabilities = @{
    mcp_servers = @()
    native_tools = @("calculator")
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/v1/agents/register -ContentType "application/json" -Body $body
```

3. Run verification:

```powershell
$verify = @{
  text = "To serialize in Pydantic, use .dict() and @validator."
  mode = "advisory"
  options = @{ persist = $true; include_evidence = $true; generate_patch = $true }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/v1/verify -ContentType "application/json" -Body $verify
```

4. Run benchmark:

```powershell
$bench = @{
  agent_ids = @("parametric_baseline", "unbounded_deep_research", "disciplined_verifier")
  suite_id = "temporal_traps_v1"
  limit_tasks = 5
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/v1/benchmarks/run -ContentType "application/json" -Body $bench
```

### 6.3 Docker Run

1. Ensure Docker Desktop is running.
2. In project root:

```powershell
docker compose -f infra/docker-compose.yml up --build
```

3. Open http://localhost:8000

## 7. Suggested Demo Script For Submission

1. Show /docs and explain architecture blocks.
2. Register 3 agents with different profiles.
3. Run verify on a known stale statement (Pydantic V1 style APIs).
4. Show claim status = STALE and the concrete suggested patch (e.g. replace `dict` with `model_dump`). Repeat with `"mode": "strict"` to show `recommended_action: block`.
5. Run the benchmark from the dashboard's Benchmark tab and show the live progress, leaderboard and answers.
6. Show how cache reduces repeated Tavily calls.
7. Reverify one claim and show claim history endpoint.

## 8. What Is Left To Reach Full Production Completion

See `NEXT_STEPS.md` sections 3 and 4. In short: ground-truth answers per task, rate limiting,
Postgres + Alembic for hosts with an ephemeral disk, a job queue for benchmarks, and encrypted
storage for provider keys.

## 9. Verification of Current Build Status

Run `python -m pytest`. The suite is offline (fixtures for Tavily, an in-process fake for the model
API) and covers verification, the LLM judge, providers, LLM-driven and external agents, background
benchmarks, API-key protection and upgrading an older database.

## 10. Quick Final Checklist Before You Submit

1. Fill .env with real Tavily key and at least one model provider key.
2. Run one full benchmark and capture output JSON/screenshots.
3. Capture at least one stale detection example with suggested patch.
4. Record a 3 to 5 minute demo video.
5. Push code to a public or hackathon-accessible repository.
6. Include this runbook in your submission package.
