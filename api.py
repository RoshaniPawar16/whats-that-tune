"""
FastAPI HTTP wrapper around the whats-that-tune agent.

Run:
    uvicorn api:app --reload

Endpoints:
    POST /session              -> create a new session
    POST /session/{id}/turn    -> one agent turn (multipart: message + optional audio)
    GET  /session/{id}/state   -> inspect current belief state
"""

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent
from session import Session

# ── Startup key check (visible in uvicorn logs) ────────────────────────────────
_api_key = os.getenv("ANTHROPIC_API_KEY", "")
print(f"[startup] ANTHROPIC_API_KEY: {'SET' if _api_key else 'NOT SET'} "
      f"(prefix={_api_key[:12]!r})")

app = FastAPI(
    title="whats-that-tune",
    description="Iterative music memory reconstruction agent. Give Mnemo clues; she'll find your song.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, Session] = {}


# ── Response models ────────────────────────────────────────────────────────────

class SessionCreated(BaseModel):
    session_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the frontend UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))


@app.post("/session", status_code=201)
def create_session() -> SessionCreated:
    """Start a new music-reconstruction session."""
    session = Session()
    _sessions[session.id] = session
    return SessionCreated(session_id=session.id)


@app.post("/session/{session_id}/turn")
async def run_turn(
    session_id: str,
    message: str = Form(""),
    audio: Optional[UploadFile] = File(default=None),
) -> dict:
    """One agent turn. Accepts multipart/form-data with an optional audio blob and text message."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if session.resolved:
        raise HTTPException(status_code=409, detail="Session is already resolved")

    tmp_path = None
    try:
        if audio and audio.filename:
            suffix = os.path.splitext(audio.filename)[1] or ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await audio.read())
                tmp_path = tmp.name

        msg = message or ("(audio clue)" if tmp_path else "")
        return run_agent(session, msg, audio_path=tmp_path)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[turn error] {exc}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/session/{session_id}/state")
def get_state(session_id: str) -> dict:
    """Return the current belief state for a session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.to_dict()
