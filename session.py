"""
Session: maintains belief state across turns.
This is the "memory" of the agent — the running probability distribution
over candidate songs, plus all clues collected so far.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    title: str
    artist: str
    confidence: float          # 0.0 – 1.0
    evidence: list[str] = field(default_factory=list)
    spotify_id: Optional[str] = None
    preview_url: Optional[str] = None

    def to_dict(self):
        return {
            "title": self.title,
            "artist": self.artist,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "spotify_id": self.spotify_id,
        }


@dataclass
class Clue:
    type: str        # "audio" | "lyric" | "mood" | "context" | "instrument" | "confirmation"
    value: str       # raw clue string or audio analysis summary
    turn: int


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.turn: int = 0
        self.candidates: list[Candidate] = []
        self.clues: list[Clue] = []
        self.messages: list[dict] = []     # full conversation history for the API
        self.last_audio_analysis: Optional[dict] = None
        self.resolved: bool = False
        self.resolved_song: Optional[Candidate] = None

    # ── Candidate management ───────────────────────────────────────────────────

    def upsert_candidate(self, title: str, artist: str, confidence: float, evidence: str):
        """Add or update a candidate. Merge evidence if it already exists."""
        for c in self.candidates:
            if c.title.lower() == title.lower() and c.artist.lower() == artist.lower():
                c.confidence = max(c.confidence, confidence)
                if evidence not in c.evidence:
                    c.evidence.append(evidence)
                return
        self.candidates.append(Candidate(title, artist, confidence, [evidence]))

    def top_candidate(self) -> Optional[Candidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)

    def add_clue(self, type: str, value: str):
        self.clues.append(Clue(type=type, value=value, turn=self.turn))

    # ── Message history ────────────────────────────────────────────────────────

    def add_message(self, role: str, content):
        self.messages.append({"role": role, "content": content})
        if role == "user":
            self.turn += 1

    # ── Agent response integration ─────────────────────────────────────────────

    def update_from_agent_response(self, response: dict):
        """Sync candidate list from agent's updated_candidates."""
        if "updated_candidates" not in response:
            return
        self.candidates = []
        for c in response["updated_candidates"]:
            self.candidates.append(Candidate(
                title=c.get("title", "Unknown"),
                artist=c.get("artist", "Unknown"),
                confidence=c.get("confidence", 0.0),
                evidence=c.get("evidence", []),
                spotify_id=c.get("spotify_id"),
                preview_url=c.get("preview_url"),
            ))
        # Sort descending by confidence
        self.candidates.sort(key=lambda c: c.confidence, reverse=True)

    def mark_resolved(self, candidate: Candidate):
        self.resolved = True
        self.resolved_song = candidate

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_id": self.id,
            "turn": self.turn,
            "candidates": [c.to_dict() for c in self.candidates],
            "clues_collected": [
                {"type": cl.type, "value": cl.value, "turn": cl.turn}
                for cl in self.clues
            ],
            "last_audio_analysis": self.last_audio_analysis,
            "resolved": self.resolved,
        }
