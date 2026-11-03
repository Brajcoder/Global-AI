from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routers.agents import router as agents_router
from apps.api.routers.benchmarks import router as benchmarks_router
from apps.api.routers.verify import router as verify_router
from packages.contracts.database import init_db

app = FastAPI(title="ChronosBench + VerityGate API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router)
app.include_router(benchmarks_router)
app.include_router(verify_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "chronosbench-veritygate"}


web_dir = Path("apps/web")
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
def root_ui():
    index = web_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "UI not found. Open API docs at /docs"}
