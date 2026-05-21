"""FastAPI entrypoint for LawGPT.

Run locally:

    cd backend
    uvicorn api.main:app --reload --port 8000

Smoke test (dev bypass — requires ``DEV_AUTH_USER=usr_demo`` in env):

    curl -N -H 'X-Dev-User: usr_demo' \
         -H 'Content-Type: application/json' \
         -X POST localhost:8000/chat \
         -d '{"query":"What is adverse possession?"}'
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.storage.db import get_engine, init_db

from .routes_chat import router as chat_router
from .routes_exam import router as exam_router
from .routes_files import router as files_router
from .routes_projects import router as projects_router
from .routes_sessions import router as sessions_router
from .routes_users import router as users_router

load_dotenv()
log = logging.getLogger("lawgpt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tables exist. For Postgres in prod, prefer Alembic migrations;
    # ``init_db`` is idempotent and safe for SQLite dev.
    engine = get_engine()
    await init_db(engine)
    log.info("DB initialised at %s", get_settings().database_url)
    yield


app = FastAPI(title="LawGPT API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/internal/warm")
def warm() -> dict:
    """Cheap end-to-end probe used by the Worker cron keep-warm.

    Touches the vector store at the SQL layer so a cold pgvector pool
    doesn't bite the first real chat request after an idle container
    spin-down. Returns ``vector_store`` so cron logs can confirm the
    backend the container booted with.
    """
    settings = get_settings()
    vector_ok = False
    try:
        # Lightweight probe: instantiate the configured store and run a
        # trivial namespace enumeration against a sentinel namespace that
        # is guaranteed not to exist. For pgvector this issues a SELECT
        # that warms the connection pool + verifies the schema.
        from src.storage.vector_store import make_vector_store

        store = make_vector_store()
        store.list_namespace("__warm_probe__")
        vector_ok = True
    except NotImplementedError:
        # Vectorize backend can't enumerate — that's still a healthy boot.
        vector_ok = True
    except Exception as exc:  # pragma: no cover — degraded path
        log.warning("warm probe: vector store unreachable (%s)", exc)
    return {
        "ok": vector_ok,
        "vector_backend": settings.vector_backend,
    }


app.include_router(chat_router)
app.include_router(projects_router)
app.include_router(sessions_router)
app.include_router(files_router)
app.include_router(exam_router)
app.include_router(users_router)
