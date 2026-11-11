from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routers.agents import router as agents_router
from apps.api.routers.benchmarks import fail_interrupted_runs
from apps.api.routers.benchmarks import router as benchmarks_router
from apps.api.routers.providers import router as providers_router
from apps.api.routers.verify import router as verify_router
from packages.contracts.database import get_session, init_db
from integrations.llm.providers import default_llm
from packages.contracts.settings import PROJECT_ROOT, get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    fail_interrupted_runs()
    yield


app = FastAPI(title="ChronosBench + VerityGate API", version="1.0.0", lifespan=lifespan)

# No cookies are used, so credentials stay off (wildcard origins + credentials is invalid).
_origins = [o.strip() for o in get_settings().cors_allow_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router)
app.include_router(benchmarks_router)
app.include_router(providers_router)
app.include_router(verify_router)


@app.get("/api/v1/health")
def health():
    settings = get_settings()
    with get_session() as session:
        llm = default_llm(session)
    return {
        "status": "ok",
        "service": "chronosbench-veritygate",
        "version": app.version,
        # What the server will actually use, so "why is everything offline?" is answerable at a glance.
        "search": "tavily" if settings.tavily_api_key else "offline_fixtures",
        "default_llm": {"provider": llm.provider, "model": llm.model} if llm else None,
        "auth_required": bool(settings.api_key),
    }


web_dir = PROJECT_ROOT / "apps" / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
def root_ui():
    index = web_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "UI not found. Open API docs at /docs"}
