"""
Tools available to the agent.
Each tool maps to a real function. The TOOL_DEFINITIONS list is passed
directly to the Anthropic API as the `tools` parameter.

Following Anthropic best practices:
  - Each tool has a precise, unambiguous description
  - Parameters use JSON Schema with clear descriptions
  - Tools are self-contained and non-overlapping
  - dispatch_tool handles all routing + error wrapping
"""

import base64
import re
import time

import librosa
import numpy as np
from typing import Any
import requests
import os

# ── Spotify helpers ───────────────────────────────────────────────────────────

_spotify_token_cache: dict = {"token": None, "expires_at": 0.0}


def _get_spotify_token() -> str | None:
    """Return a valid Spotify client-credentials token, refreshing if expired."""
    global _spotify_token_cache
    if _spotify_token_cache["token"] and time.time() < _spotify_token_cache["expires_at"]:
        return _spotify_token_cache["token"]

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        _spotify_token_cache["token"] = data["access_token"]
        _spotify_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
        return _spotify_token_cache["token"]
    except Exception:
        return None


def _decade_to_year_filter(decade: str) -> str:
    """Convert a decade description to a Spotify year filter string.

    Examples:
        '1990s'       → 'year:1990-1999'
        'early 2000s' → 'year:2000-2004'
        'late 80s'    → 'year:1985-1989'
    """
    d = decade.lower().strip()
    modifier = "early" if "early" in d else ("late" if "late" in d else None)

    m4 = re.search(r"\b(\d{4})\b", d)
    m2 = re.search(r"\b(\d{2})s\b", d)

    if m4:
        start = (int(m4.group(1)) // 10) * 10
    elif m2:
        n = int(m2.group(1))
        start = (1900 if n >= 20 else 2000) + n
    else:
        return ""

    if modifier == "early":
        return f"year:{start}-{start + 4}"
    if modifier == "late":
        return f"year:{start + 5}-{start + 9}"
    return f"year:{start}-{start + 9}"


# ── Tool definitions (passed to API) ──────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "analyse_audio",
        "description": (
            "Analyse a hummed or sung audio recording. Extracts: dominant pitch sequence, "
            "tempo estimate, key signature estimate, chroma features, and a human-readable "
            "melodic description. ALWAYS call this first when an audio_path is present. "
            "Do NOT call search_melody before calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the recorded audio file (.wav or .mp3)"
                },
                "duration_hint": {
                    "type": "number",
                    "description": "Expected duration in seconds. Used to set analysis window. Default 10."
                }
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "search_melody",
        "description": (
            "Search for songs matching a melodic contour described as a pitch sequence. "
            "Uses chromagram DTW matching against the reference corpus. "
            "Call AFTER analyse_audio has returned pitch data. "
            "Do NOT call this without a pitch_sequence — use search_context instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pitch_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of note names in order, e.g. ['C4', 'E4', 'G4', 'A4']. "
                                   "Use 'X' for uncertain pitches."
                },
                "tempo_bpm": {
                    "type": "number",
                    "description": "Estimated tempo in BPM. Can be approximate."
                },
                "key": {
                    "type": "string",
                    "description": "Estimated key, e.g. 'C major' or 'A minor'. Optional."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of candidates to return. Default 5."
                }
            },
            "required": ["pitch_sequence"]
        }
    },
    {
        "name": "search_lyrics",
        "description": (
            "Search for songs matching lyric fragments. Works with partial, misheard, "
            "or approximate lyrics — even 2-3 words. Call when the user provides any "
            "text they associate with the song. "
            "Do NOT call with an empty string — wait for at least one word."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lyric_fragment": {
                    "type": "string",
                    "description": "The lyric words the user remembers. Can be approximate or partial."
                },
                "context_hint": {
                    "type": "string",
                    "description": "Any extra context: decade, language, genre, mood. Optional."
                }
            },
            "required": ["lyric_fragment"]
        }
    },
    {
        "name": "search_context",
        "description": (
            "Search for songs using non-audio contextual clues: decade, genre, mood, "
            "instrumentation, cultural context (film, ad, TV show), or emotional memory. "
            "Use when the user can describe the song but cannot hum it clearly. "
            "Do NOT use as a substitute for search_melody when audio is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Free-text description of the song's feel, context, or characteristics."
                },
                "decade": {
                    "type": "string",
                    "description": "Approximate decade, e.g. '1990s' or 'early 2000s'. Optional."
                },
                "genre_hint": {
                    "type": "string",
                    "description": "Genre or style hint, e.g. 'rock', 'film score', 'R&B'. Optional."
                },
                "instruments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Instruments the user remembers, e.g. ['piano', 'strings']. Optional."
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "play_candidate",
        "description": (
            "Queue a song preview for the user to listen to and confirm. "
            "Only call when top candidate confidence > 0.7. "
            "Returns a preview URL and prompts the user to confirm or reject. "
            "Do NOT call speculatively — only when you're reasonably confident."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Song title"
                },
                "artist": {
                    "type": "string",
                    "description": "Artist name"
                },
                "spotify_id": {
                    "type": "string",
                    "description": "Spotify track ID if available. Optional."
                }
            },
            "required": ["title", "artist"]
        }
    }
]


# ── Tool implementations ───────────────────────────────────────────────────────

def _analyse_audio(params: dict, session) -> dict:
    """
    Extract musical features from a hummed/sung audio file.
    Uses librosa + basic-pitch-style pitch tracking.
    """
    audio_path = params["audio_path"]
    duration_hint = params.get("duration_hint", 10.0)

    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=duration_hint, mono=True)

        # Pitch tracking via pyin (more robust for humming than YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
        )

        # Convert Hz to note names, filter unvoiced frames
        notes = []
        for freq, voiced in zip(f0, voiced_flag):
            if voiced and freq and not np.isnan(freq):
                note = librosa.hz_to_note(freq)
                if not notes or notes[-1] != note:
                    notes.append(note)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.round(tempo, 1)) if np.isscalar(tempo) else float(np.round(tempo[0], 1))

        # Key estimation via chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = note_names[int(np.argmax(chroma_mean))]

        result = {
            "pitch_sequence": notes[:32],    # cap at 32 notes for context efficiency
            "tempo_bpm": tempo_val,
            "estimated_key": estimated_key,
            "duration_seconds": round(len(y) / sr, 2),
            "note_count": len(notes),
            "melodic_description": (
                f"A {round(len(y)/sr,1)}s fragment in approximately {estimated_key}. "
                f"Tempo ~{tempo_val} BPM. {len(notes)} distinct pitch transitions detected."
            )
        }

        session.last_audio_analysis = result
        session.add_clue("audio", result["melodic_description"])
        return result

    except Exception as e:
        return {"error": str(e), "pitch_sequence": [], "tempo_bpm": None, "estimated_key": None}


def _search_melody(params: dict, session) -> dict:
    """
    Match a pitch sequence against the chromagram corpus index using DTW.
    Requires a built corpus index at CHROMA_INDEX_PATH (default: corpus/chroma_index.pkl).
    Build the index with: python scripts/index_library.py (see README).
    """
    pitch_sequence = params["pitch_sequence"]
    top_k = params.get("top_k", 5)
    session.add_clue("audio", f"Melody search: {' '.join(pitch_sequence[:10])}")

    index_path = os.getenv("CHROMA_INDEX_PATH", "corpus/chroma_index.pkl")
    if not os.path.exists(index_path):
        return {
            "status": "no corpus index found",
            "candidates": [],
            "note": (
                f"Build an index first: set CHROMA_INDEX_PATH or place corpus at {index_path}. "
                "See README for index build options."
            ),
        }

    try:
        from matcher import ChromaMatcher
        matcher = ChromaMatcher.load(index_path)
        candidates = matcher.query(pitch_sequence, top_k=top_k)
        for c in candidates:
            session.upsert_candidate(c["title"], c["artist"], c["confidence"], "melody_dtw")
        return {"candidates": candidates}
    except Exception as e:
        return {"error": str(e), "candidates": []}


def _search_lyrics(params: dict, session) -> dict:
    """
    Lyrics search via Genius API (or your preferred provider).
    Requires GENIUS_API_TOKEN env var.
    """
    fragment = params["lyric_fragment"]
    context = params.get("context_hint", "")
    session.add_clue("lyric", fragment)

    token = os.getenv("GENIUS_API_TOKEN")
    if not token:
        return {"error": "GENIUS_API_TOKEN not set", "candidates": []}

    try:
        query = f"{fragment} {context}".strip()
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(
            "https://api.genius.com/search",
            params={"q": query, "per_page": 5},
            headers=headers,
            timeout=5,
        )
        r.raise_for_status()
        hits = r.json()["response"]["hits"]
        return {
            "candidates": [
                {
                    "title": h["result"]["title"],
                    "artist": h["result"]["primary_artist"]["name"],
                    "confidence": 0.4,  # base confidence; agent refines this
                    "source": "genius_lyrics",
                }
                for h in hits
            ]
        }
    except Exception as e:
        return {"error": str(e), "candidates": []}


def _search_context(params: dict, session) -> dict:
    """
    Context-based search via Spotify search API, filtered by decade/genre.
    Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in environment.
    """
    description = params["description"]
    decade = params.get("decade", "")
    genre = params.get("genre_hint", "")
    instruments = params.get("instruments", [])

    session.add_clue("context", f"{description} | decade:{decade} | genre:{genre}")

    token = _get_spotify_token()
    if not token:
        return {
            "error": "Spotify credentials not set — add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env",
            "candidates": [],
        }

    # Build Spotify search query
    parts = [description]
    if instruments:
        parts.extend(instruments)
    if genre:
        parts.append(f"genre:{genre}")
    year_filter = _decade_to_year_filter(decade) if decade else ""
    if year_filter:
        parts.append(year_filter)

    query = " ".join(parts)

    try:
        r = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 5},
            timeout=5,
        )
        r.raise_for_status()
        items = r.json()["tracks"]["items"]
        candidates = [
            {
                "title": t["name"],
                "artist": t["artists"][0]["name"],
                "spotify_id": t["id"],
                "preview_url": t.get("preview_url"),
                "confidence": 0.35,  # base; agent updates from context reasoning
                "source": "spotify_context",
            }
            for t in items
        ]
        for c in candidates:
            session.upsert_candidate(c["title"], c["artist"], c["confidence"], "context_search")
        return {"candidates": candidates, "query": query}
    except Exception as e:
        return {"error": str(e), "candidates": []}


def _play_candidate(params: dict, session) -> dict:
    """
    Fetch a Spotify 30s preview URL for the candidate and surface it to the user.
    Falls back to the Spotify search page URL if no preview is available.
    """
    title = params["title"]
    artist = params["artist"]
    spotify_id = params.get("spotify_id")

    fallback_url = (
        f"https://open.spotify.com/search/"
        f"{requests.utils.quote(f'{title} {artist}')}"
    )

    token = _get_spotify_token()
    if not token:
        return {
            "title": title,
            "artist": artist,
            "preview_url": fallback_url,
            "preview_type": "search_fallback",
            "instruction": "Play this for the user and ask: 'Is this it?'",
        }

    try:
        if spotify_id:
            r = requests.get(
                f"https://api.spotify.com/v1/tracks/{spotify_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            r.raise_for_status()
            track = r.json()
        else:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": f"track:{title} artist:{artist}", "type": "track", "limit": 1},
                timeout=5,
            )
            r.raise_for_status()
            items = r.json()["tracks"]["items"]
            if not items:
                return {
                    "title": title,
                    "artist": artist,
                    "preview_url": fallback_url,
                    "preview_type": "search_fallback",
                    "instruction": "Play this for the user and ask: 'Is this it?'",
                }
            track = items[0]

        preview_url = track.get("preview_url") or fallback_url
        preview_type = "spotify_preview" if track.get("preview_url") else "search_fallback"
        spotify_url = track.get("external_urls", {}).get("spotify", fallback_url)

        return {
            "title": track.get("name", title),
            "artist": track["artists"][0]["name"] if track.get("artists") else artist,
            "spotify_id": track.get("id", spotify_id),
            "preview_url": preview_url,
            "spotify_url": spotify_url,
            "preview_type": preview_type,
            "instruction": "Play this for the user and ask: 'Is this it?'",
        }
    except Exception as e:
        return {
            "title": title,
            "artist": artist,
            "preview_url": fallback_url,
            "preview_type": "search_fallback",
            "error": str(e),
            "instruction": "Play this for the user and ask: 'Is this it?'",
        }


# ── Dispatcher ────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "analyse_audio": _analyse_audio,
    "search_melody": _search_melody,
    "search_lyrics": _search_lyrics,
    "search_context": _search_context,
    "play_candidate": _play_candidate,
}


def dispatch_tool(name: str, params: dict, session) -> Any:
    """Route a tool call to its implementation. Always returns a dict."""
    fn = TOOL_MAP.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(params, session)
    except Exception as e:
        return {"error": f"Tool {name} failed: {str(e)}"}
