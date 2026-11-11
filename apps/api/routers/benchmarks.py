from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlmodel import Session, select

from agents.base import AgentContext, AgentRun, Profile
from agents.runtime import resolve_profile, run_http_agent, run_profile
from apps.api.security import require_api_key
from apps.api.services import verify_text
from integrations.llm.client import LLMConfig
from integrations.llm.providers import resolve_llm
from integrations.tavily.client import TavilyClient
from packages.contracts.database import (
    AgentManifestDB,
    AgentTrajectoryDB,
    BenchmarkRunDB,
    BenchmarkScoreDB,
    get_session,
    session_dep,
    utcnow,
)
from packages.contracts.schemas import (
    BenchmarkAgentScore,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    LLMOverride,
    VerifyRequest,
)
from packages.contracts.settings import PROJECT_ROOT, get_settings
from packages.scoring_engine.scoring import chronos_score, eff_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["benchmarks"])

SUITES = {
    "temporal_traps_v1": PROJECT_ROOT / "benchmarks" / "datasets" / "temporal_traps.json",
}

# Strong references to running background benchmarks, so they are not garbage collected mid-run.
_background_tasks: set[asyncio.Task] = set()


@dataclass
class AgentPlan:
    """Everything needed to run one agent, detached from the request's DB session."""

    agent_id: str
    model_name: str
    profile: Profile
    endpoint: str
    llm: LLMConfig | None
    system_prompt: str
    instruction_template: str
    params_json: str
    tools_json: str


def _load_dataset(suite_id: str) -> list[dict]:
    path = SUITES.get(suite_id)
    if path is None:
        raise HTTPException(status_code=400, detail=f"Unknown suite_id: {suite_id}. Known: {sorted(SUITES)}")
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Dataset file missing for suite {suite_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_agent(session: Session, agent_id: str) -> AgentPlan:
    manifest = session.get(AgentManifestDB, agent_id)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    endpoint = manifest.endpoint_or_command if manifest.endpoint_or_command.startswith(("http://", "https://")) else ""
    llm = None
    if not endpoint:
        llm = resolve_llm(
            session,
            manifest.provider,
            manifest.model_name,
            manifest.base_url,
            manifest.api_key,
            temperature=manifest.temperature,
            top_p=manifest.top_p,
            max_tokens=manifest.max_tokens,
        )
    return AgentPlan(
        agent_id=manifest.id,
        model_name=llm.model if llm else manifest.model_name,
        profile=resolve_profile(manifest.profile, manifest.id),
        endpoint=endpoint,
        llm=llm,
        system_prompt=manifest.system_prompt,
        instruction_template=manifest.instruction_template,
        params_json=json.dumps(
            {"temperature": manifest.temperature, "top_p": manifest.top_p, "max_tokens": manifest.max_tokens}
        ),
        tools_json=manifest.mcp_servers_json,
    )


@router.get("/benchmarks/suites")
def list_suites():
    out = []
    for suite_id, path in SUITES.items():
        tasks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        cohorts: dict[str, int] = {}
        for task in tasks:
            cohorts[task["cohort"]] = cohorts.get(task["cohort"], 0) + 1
        out.append({"suite_id": suite_id, "tasks": len(tasks), "cohorts": cohorts})
    return out


@router.get("/benchmarks")
def list_benchmarks(limit: int = 20, session: Session = Depends(session_dep)):
    limit = max(1, min(limit, 100))
    runs = session.exec(select(BenchmarkRunDB).order_by(BenchmarkRunDB.started_at.desc()).limit(limit)).all()
    return [_run_response(session, run).model_dump() | {"started_at": run.started_at, "completed_at": run.completed_at} for run in runs]


@router.post("/benchmarks/run", response_model=BenchmarkRunResponse, dependencies=[Depends(require_api_key)])
async def run_benchmark(payload: BenchmarkRunRequest, response: Response, session: Session = Depends(session_dep)):
    # Validate everything up front so a bad request never leaves a half-created run behind.
    tasks = _load_dataset(payload.suite_id)
    if payload.limit_tasks:
        tasks = tasks[: payload.limit_tasks]
    agent_ids = list(dict.fromkeys(payload.agent_ids))  # drop duplicates, keep order
    plans = [_plan_agent(session, agent_id) for agent_id in agent_ids]
    judge = payload.judge_llm
    if judge is not None and resolve_llm(session, judge.provider, judge.model, judge.base_url, judge.api_key) is None:
        raise HTTPException(status_code=400, detail=f"Judge LLM provider '{judge.provider or 'default'}' is not configured.")

    run_id = f"bm_{uuid.uuid4().hex[:12]}"
    session.add(BenchmarkRunDB(id=run_id, suite_id=payload.suite_id, status="running", tasks_total=len(tasks) * len(plans)))
    session.commit()

    if payload.background:
        task = asyncio.create_task(_execute_detached(run_id, plans, tasks, judge))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        response.status_code = 202
        return _run_response(session, session.get(BenchmarkRunDB, run_id))

    ok = await _execute(session, run_id, plans, tasks, judge)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Benchmark {run_id} failed; see server logs.")
    return _run_response(session, session.get(BenchmarkRunDB, run_id))


async def _execute_detached(run_id: str, plans: list[AgentPlan], tasks: list[dict], judge: LLMOverride | None) -> None:
    with get_session() as session:
        await _execute(session, run_id, plans, tasks, judge)


async def _execute(session: Session, run_id: str, plans: list[AgentPlan], tasks: list[dict], judge: LLMOverride | None) -> bool:
    """Run every agent over every task. Returns False (and marks the run failed) on an unexpected error."""
    try:
        sources = set()
        for plan in plans:
            score = await _run_agent(session, run_id, plan, tasks, judge)
            sources.add(score.telemetry_source)
            session.add(BenchmarkScoreDB(id=f"bs_{uuid.uuid4().hex[:12]}", benchmark_id=run_id, **score.model_dump()))
            session.commit()
    except Exception as exc:
        logger.exception("Benchmark %s failed", run_id)
        session.rollback()
        run = session.get(BenchmarkRunDB, run_id)
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"[:500]
        run.completed_at = utcnow()
        session.add(run)
        session.commit()
        return False

    run = session.get(BenchmarkRunDB, run_id)
    run.status = "completed"
    run.telemetry_source = sources.pop() if len(sources) == 1 else "mixed"
    run.completed_at = utcnow()
    session.add(run)
    session.commit()
    return True


async def _run_task(plan: AgentPlan, task: dict, tavily: TavilyClient) -> AgentRun:
    settings = get_settings()
    if plan.endpoint:
        payload = {"task_id": task["id"], "prompt": task["prompt"], "agent_id": plan.agent_id, "model": plan.model_name}
        return await run_http_agent(plan.endpoint, payload, timeout=settings.llm_timeout_seconds)
    ctx = AgentContext(
        llm=plan.llm,
        search=tavily.search,
        system_prompt=plan.system_prompt,
        instruction_template=plan.instruction_template,
        max_steps=settings.max_agent_steps_per_task,
    )
    return await run_profile(plan.profile, ctx, task["prompt"])


def focus_coverage(output: str, expected_focus: list[str]) -> float:
    """Share (0-100) of the task's expected focus terms that the answer mentions."""
    if not expected_focus:
        return 0.0
    text = _normalize(output)
    hits = sum(1 for term in expected_focus if _normalize(term) in text)
    return 100.0 * hits / len(expected_focus)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", text.lower()).split())


async def _run_agent(session: Session, run_id: str, plan: AgentPlan, tasks: list[dict], judge: LLMOverride | None) -> BenchmarkAgentScore:
    tavily = TavilyClient()
    truth_scores: list[float] = []
    eff_scores: list[float] = []
    coverages: list[float] = []
    sources: set[str] = set()
    total_tokens = total_latency = failed = 0

    for task in tasks:
        run = await _run_task(plan, task, tavily)
        loop_ratio = run.loop_count / max(1, run.steps)
        e_score = eff_score(run.tokens, run.latency_ms, loop_ratio)
        coverage = focus_coverage(run.output, task.get("expected_focus", []))
        sources.add(run.telemetry_source)
        total_tokens += run.tokens
        total_latency += run.latency_ms

        traj_id = f"traj_{uuid.uuid4().hex[:12]}"
        trajectory = AgentTrajectoryDB(
            id=traj_id,
            benchmark_id=run_id,
            agent_id=plan.agent_id,
            task_id=task["id"],
            observed_model=run.model or plan.model_name,
            observed_params_json=plan.params_json,
            observed_tools_json=plan.tools_json,
            invoked_tool_calls_json=json.dumps(run.tool_calls),
            step_count=run.steps,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            latency_ms=run.latency_ms,
            loop_count=run.loop_count,
            raw_output=run.output,
            eff_score=round(e_score, 2),
            focus_coverage=round(coverage, 2),
            telemetry_source=run.telemetry_source,
            error=run.error,
        )
        session.add(trajectory)

        truth = 0.0
        if run.error or not run.output:
            # A task the agent could not answer scores zero truth; the run carries on.
            failed += 1
            logger.warning("Agent %s failed task %s: %s", plan.agent_id, task["id"], run.error)
        else:
            verify_req = VerifyRequest(
                text=run.output,
                mode="advisory",
                context={
                    "domain": "software",
                    "locale": "en",
                    "run_id": run_id,
                    "agent_id": plan.agent_id,
                    "model_used": run.model or plan.model_name,
                },
                options={"max_claims": 8, "include_evidence": True, "generate_patch": True, "persist": True, "llm": judge},
            )
            truth = (await verify_text(verify_req, session, trajectory_id=traj_id)).truth_score

        trajectory.truth_score = truth
        truth_scores.append(truth)
        eff_scores.append(e_score)
        coverages.append(coverage)

        bench = session.get(BenchmarkRunDB, run_id)
        bench.tasks_done += 1
        session.add(bench)
        session.add(trajectory)
        session.commit()

    avg_truth = round(sum(truth_scores) / max(1, len(truth_scores)), 2)
    avg_eff = round(sum(eff_scores) / max(1, len(eff_scores)), 2)
    return BenchmarkAgentScore(
        agent_id=plan.agent_id,
        model_name=plan.model_name,
        truth_score=avg_truth,
        eff_score=avg_eff,
        chronos_score=round(chronos_score(avg_truth, avg_eff), 2),
        focus_coverage=round(sum(coverages) / max(1, len(coverages)), 2),
        tokens_used=total_tokens,
        latency_ms=total_latency,
        tasks_failed=failed,
        telemetry_source=sources.pop() if len(sources) == 1 else "mixed",
    )


def _leaderboard(session: Session, benchmark_id: str) -> list[BenchmarkAgentScore]:
    rows = session.exec(select(BenchmarkScoreDB).where(BenchmarkScoreDB.benchmark_id == benchmark_id)).all()
    scores = [BenchmarkAgentScore(**row.model_dump(exclude={"id", "benchmark_id"})) for row in rows]
    return sorted(scores, key=lambda s: s.chronos_score, reverse=True)


def _run_response(session: Session, run: BenchmarkRunDB) -> BenchmarkRunResponse:
    return BenchmarkRunResponse(
        benchmark_id=run.id,
        status=run.status,
        suite_id=run.suite_id,
        tasks_total=run.tasks_total,
        tasks_done=run.tasks_done,
        leaderboard=_leaderboard(session, run.id),
        telemetry_source=run.telemetry_source,
        error=run.error,
    )


@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, session: Session = Depends(session_dep)):
    run = session.get(BenchmarkRunDB, benchmark_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    trajectories = session.exec(select(AgentTrajectoryDB).where(AgentTrajectoryDB.benchmark_id == benchmark_id)).all()
    return {
        **_run_response(session, run).model_dump(),
        "run": run.model_dump(),
        "trajectories": [t.model_dump() for t in trajectories],
    }


def fail_interrupted_runs() -> None:
    """On startup, mark runs still 'running' as failed: their background task died with the old process."""
    with get_session() as session:
        for run in session.exec(select(BenchmarkRunDB).where(BenchmarkRunDB.status == "running")).all():
            run.status = "failed"
            run.error = "Interrupted by a server restart."
            run.completed_at = utcnow()
            session.add(run)
        session.commit()
