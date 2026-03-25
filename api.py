"""
FastAPI HTTP wrapper around the whats-that-tune agent.

Run:
    uvicorn api:app --reload

Endpoints:
    POST /session              → create a new session
    POST /session/{id}/turn    → run one agent turn
    GET  /session/{id}/state   → inspect current belief state
"""

from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent import run_agent  # noqa: E402 (after load_dotenv)
from session import Session  # noqa: E402

app = FastAPI(
    title="whats-that-tune",
    description="Iterative music memory reconstruction agent. Give Mnemo clues; she'll find your song.",
    version="0.1.0",
)

_sessions: dict[str, Session] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class TurnRequest(BaseModel):
    message: str
    audio_path: Optional[str] = None


class SessionCreated(BaseModel):
    session_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/session", response_model=SessionCreated, status_code=201)
def create_session() -> dict:
    """Start a new music-reconstruction session."""
    session = Session()
    _sessions[session.id] = session
    return {"session_id": session.id}


@app.post("/session/{session_id}/turn")
def run_turn(session_id: str, body: TurnRequest) -> dict:
    """Submit a user message (and optional audio path) for one agent turn.

    Returns the agent's JSON response with updated_candidates, next_question,
    message_to_user, and confidence_summary.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if session.resolved:
        raise HTTPException(status_code=409, detail="Session is already resolved")

    return run_agent(session, body.message, audio_path=body.audio_path)


@app.get("/session/{session_id}/state")
def get_state(session_id: str) -> dict:
    """Return the current belief state for a session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.to_dict()
