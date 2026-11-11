from __future__ import annotations

import importlib
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from apps.api.services import verify_text
from integrations.bifrost_gateway.telemetry import estimate_telemetry
from packages.contracts.database import AgentManifestDB, AgentTrajectoryDB, BenchmarkRunDB, session_dep, utcnow
from packages.contracts.schemas import BenchmarkAgentScore, BenchmarkRunRequest, BenchmarkRunResponse, VerifyRequest
from packages.contracts.settings import PROJECT_ROOT
from packages.scoring_engine.scoring import chronos_score, eff_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["benchmarks"])

AGENT_MODULES = {
    "parametric_baseline": "agents.parametric_baseline",
    "unbounded_deep_research": "agents.unbounded_deep_research",
    "disciplined_verifier": "agents.disciplined_verifier",
}

SUITES = {
    "temporal_traps_v1": PROJECT_ROOT / "benchmarks" / "datasets" / "temporal_traps.json",
}


def _load_dataset(suite_id: str) -> list[dict]:
    path = SUITES.get(suite_id)
    if path is None:
        raise HTTPException(status_code=400, detail=f"Unknown suite_id: {suite_id}. Known: {sorted(SUITES)}")
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Dataset file missing for suite {suite_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/benchmarks/run", response_model=BenchmarkRunResponse)
async def run_benchmark(payload: BenchmarkRunRequest, session: Session = Depends(session_dep)):
    # Validate everything up front so a bad request never leaves a half-created run behind.
    tasks = _load_dataset(payload.suite_id)
    if payload.limit_tasks:
        tasks = tasks[: payload.limit_tasks]

    agents: list[tuple[AgentManifestDB, object]] = []
    for agent_id in payload.agent_ids:
        manifest = session.get(AgentManifestDB, agent_id)
        if not manifest:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        module_name = AGENT_MODULES.get(agent_id)
        if not module_name:
            raise HTTPException(status_code=400, detail=f"No runnable module mapped for {agent_id}")
        agents.append((manifest, importlib.import_module(module_name)))

    run_id = f"bm_{uuid.uuid4().hex[:12]}"
    run = BenchmarkRunDB(id=run_id, suite_id=payload.suite_id, status="running")
    session.add(run)
    session.commit()

    try:
        rows = [await _run_agent(session, run_id, manifest, module, tasks) for manifest, module in agents]
    except Exception:
        logger.exception("Benchmark %s failed", run_id)
        session.rollback()
        run.status = "failed"
        run.completed_at = utcnow()
        session.add(run)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Benchmark {run_id} failed; see server logs.")

    run.status = "completed"
    run.completed_at = utcnow()
    session.add(run)
    session.commit()

    rows = sorted(rows, key=lambda r: r.chronos_score, reverse=True)
    return BenchmarkRunResponse(benchmark_id=run_id, status="completed", leaderboard=rows)


async def _run_agent(session: Session, run_id: str, manifest: AgentManifestDB, module, tasks: list[dict]) -> BenchmarkAgentScore:
    truth_scores: list[float] = []
    eff_scores: list[float] = []
    total_tokens = 0
    total_latency = 0

    for task in tasks:
        output = module.run_task(task["prompt"])
        telemetry = estimate_telemetry(
            output,
            model_name=manifest.model_name,
            temperature=manifest.temperature,
            top_p=manifest.top_p,
            max_tokens=manifest.max_tokens,
        )

        tokens = telemetry.prompt_tokens + telemetry.completion_tokens
        total_tokens += tokens
        total_latency += telemetry.latency_ms
        loop_ratio = telemetry.loop_count / max(1, telemetry.step_count)
        e_score = eff_score(tokens, telemetry.latency_ms, loop_ratio)

        traj_id = f"traj_{uuid.uuid4().hex[:12]}"
        session.add(
            AgentTrajectoryDB(
                id=traj_id,
                benchmark_id=run_id,
                agent_id=manifest.id,
                task_id=task["id"],
                observed_model=manifest.model_name,
                observed_params_json=json.dumps(
                    {"temperature": manifest.temperature, "top_p": manifest.top_p, "max_tokens": manifest.max_tokens}
                ),
                observed_tools_json=manifest.mcp_servers_json,
                invoked_tool_calls_json="[]",
                step_count=telemetry.step_count,
                prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens,
                latency_ms=telemetry.latency_ms,
                loop_count=telemetry.loop_count,
                raw_output=output,
                eff_score=round(e_score, 2),
            )
        )

        verify_req = VerifyRequest(
            text=output,
            mode="advisory",
            context={
                "domain": "software",
                "locale": "en",
                "run_id": run_id,
                "agent_id": manifest.id,
                "model_used": manifest.model_name,
            },
            options={"max_claims": 8, "include_evidence": True, "generate_patch": True, "persist": True},
        )
        verification = await verify_text(verify_req, session, trajectory_id=traj_id)
        truth_scores.append(verification.truth_score)
        eff_scores.append(e_score)

    avg_truth = round(sum(truth_scores) / max(1, len(truth_scores)), 2)
    avg_eff = round(sum(eff_scores) / max(1, len(eff_scores)), 2)
    return BenchmarkAgentScore(
        agent_id=manifest.id,
        truth_score=avg_truth,
        eff_score=avg_eff,
        chronos_score=round(chronos_score(avg_truth, avg_eff), 2),
        tokens_used=total_tokens,
        latency_ms=total_latency,
    )


@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, session: Session = Depends(session_dep)):
    run = session.get(BenchmarkRunDB, benchmark_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    trajectories = session.exec(select(AgentTrajectoryDB).where(AgentTrajectoryDB.benchmark_id == benchmark_id)).all()
    return {
        "run": run.model_dump(),
        "trajectories": [t.model_dump() for t in trajectories],
    }
