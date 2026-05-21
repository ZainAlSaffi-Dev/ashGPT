"""Chat endpoint — streams agent pipeline output as Server-Sent Events.

Event types emitted:
  * ``node``           — when a graph node starts/finishes (node_trace step)
  * ``answer_chunk``   — incremental final-answer text (Phase 1: single full-blob event)
  * ``sources``        — retrieved chunk metadata
  * ``verification``   — verification report
  * ``done``           — terminal marker with full payload
  * ``error``          — failure
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.agent.graph import run_query
from src.config import SYNTHESIS_MODEL
from src.storage.db import Message, Session, User

from .deps import current_user, db_session
from .schemas import ChatRequest

router = APIRouter(tags=["chat"])
log = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def _ensure_session(
    db: AsyncSession, user: User, session_id: str | None, fallback_title: str
) -> Session:
    if session_id:
        sess = (
            await db.execute(
                select(Session).where(Session.id == session_id, Session.user_id == user.id)
            )
        ).scalar_one_or_none()
        if sess is None:
            raise HTTPException(404, "session not found")
        return sess
    sess = Session(user_id=user.id, title=fallback_title[:80] or "New chat")
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    return sess


async def _load_history(db: AsyncSession, session_id: str) -> list[dict[str, str]]:
    rows = (
        await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
        )
    ).scalars().all()
    return [{"role": m.role, "content": m.content} for m in rows]


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> EventSourceResponse:
    session = await _ensure_session(db, user, body.session_id, fallback_title=body.query)
    history = await _load_history(db, session.id)

    user_msg = Message(
        session_id=session.id, user_id=user.id, role="user", content=body.query
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)
    user_msg_id = user_msg.id

    # Capture vars for the closure
    user_id = user.id
    session_id = session.id
    week_filter = body.week_filter
    query = body.query

    async def stream() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        started = time.time()
        try:
            yield _sse("node", {"node": "start", "session_id": session_id})
            # run_query is synchronous (LangGraph .invoke). Push to a threadpool
            # so the FastAPI event loop stays responsive.
            result = await loop.run_in_executor(
                None,
                lambda: run_query(
                    query=query,
                    week_filter=week_filter,
                    chat_history=history,
                    user_id=user_id,
                ),
            )
            latency_ms = int((time.time() - started) * 1000)

            for step in result.get("node_trace", []) or []:
                yield _sse("node", {"node": step})

            sources = [
                {
                    "source": d.get("source"),
                    "doc_type": d.get("doc_type"),
                    "week": d.get("week"),
                    "snippet": (d.get("content") or "")[:280],
                }
                for d in (result.get("retrieved_texts") or [])
                + (result.get("retrieved_slides") or [])
            ]
            yield _sse("sources", {"sources": sources})

            if result.get("irac_analysis"):
                yield _sse("irac", {"irac": result["irac_analysis"]})
            if result.get("mermaid_diagram"):
                yield _sse("mermaid", {"diagram": result["mermaid_diagram"]})
            if result.get("verification_report"):
                yield _sse("verification", {"report": result["verification_report"]})
            if result.get("timings"):
                yield _sse("timings", {"timings": result["timings"]})
            overflow = result.get("chat_history_overflow")
            if overflow:
                yield _sse("history_overflow", overflow)
            memory_telemetry = result.get("memory_telemetry")
            if memory_telemetry:
                yield _sse("memory", memory_telemetry)

            final = result.get("final_answer", "")
            yield _sse("answer_chunk", {"text": final})

            # Stamp the synthesis model + escalation flag onto the verification
            # blob so the message log captures which model produced the final
            # answer. Verification dict may be None (general intent path).
            verification = dict(result.get("verification_report") or {})
            escalated_to = result.get("escalated_to")
            verification["synthesis_model"] = escalated_to or SYNTHESIS_MODEL
            verification["escalated"] = bool(escalated_to)
            if escalated_to:
                verification["escalated_from"] = result.get("escalated_from")

            # Persist assistant message (await directly — bg task could lose the row).
            assistant_msg = Message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=final,
                intent=result.get("intent"),
                retrieved_chunk_ids=[s.get("source") for s in sources if s.get("source")],
                sources=sources,
                irac=result.get("irac_analysis"),
                mermaid=result.get("mermaid_diagram"),
                latency_ms=latency_ms,
                verification=verification,
            )
            db.add(assistant_msg)
            await db.commit()

            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "intent": result.get("intent"),
                    "latency_ms": latency_ms,
                    "final_answer": final,
                },
            )
        except Exception as e:
            # Graph failed mid-stream → roll back the orphan user message so
            # the conversation history stays coherent for the next turn.
            log.exception("chat stream failed; deleting orphan user message %s", user_msg_id)
            try:
                await db.execute(
                    Message.__table__.delete().where(Message.id == user_msg_id)
                )
                await db.commit()
            except Exception:  # pragma: no cover — last-ditch cleanup
                log.exception("orphan rollback failed; manual cleanup required")
            yield _sse("error", {"detail": str(e)})

    return EventSourceResponse(stream())
