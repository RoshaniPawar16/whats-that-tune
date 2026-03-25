"""
Tests for session.py — belief-state management.

Run from the project root:
    pytest tests/
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from session import Session, Candidate


# ── upsert_candidate ───────────────────────────────────────────────────────────

def test_upsert_candidate_adds_new():
    s = Session()
    s.upsert_candidate("Bohemian Rhapsody", "Queen", 0.5, "lyric match")
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert c.title == "Bohemian Rhapsody"
    assert c.artist == "Queen"
    assert c.confidence == 0.5
    assert c.evidence == ["lyric match"]


def test_upsert_candidate_merges_existing():
    s = Session()
    s.upsert_candidate("Bohemian Rhapsody", "Queen", 0.5, "lyric match")
    s.upsert_candidate("Bohemian Rhapsody", "Queen", 0.8, "audio match")
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert c.confidence == 0.8  # takes the higher value
    assert len(c.evidence) == 2


def test_upsert_candidate_does_not_duplicate_evidence():
    s = Session()
    s.upsert_candidate("Song A", "Artist A", 0.4, "same clue")
    s.upsert_candidate("Song A", "Artist A", 0.6, "same clue")
    assert s.candidates[0].evidence == ["same clue"]


def test_upsert_candidate_case_insensitive_merge():
    s = Session()
    s.upsert_candidate("bohemian rhapsody", "queen", 0.5, "first")
    s.upsert_candidate("Bohemian Rhapsody", "Queen", 0.7, "second")
    assert len(s.candidates) == 1
    assert s.candidates[0].confidence == 0.7


def test_upsert_candidate_distinct_songs_stay_separate():
    s = Session()
    s.upsert_candidate("Song A", "Artist A", 0.5, "clue1")
    s.upsert_candidate("Song B", "Artist B", 0.6, "clue2")
    assert len(s.candidates) == 2


# ── top_candidate ──────────────────────────────────────────────────────────────

def test_top_candidate_returns_highest_confidence():
    s = Session()
    s.upsert_candidate("Song A", "Artist A", 0.3, "c1")
    s.upsert_candidate("Song B", "Artist B", 0.8, "c2")
    s.upsert_candidate("Song C", "Artist C", 0.5, "c3")
    top = s.top_candidate()
    assert top is not None
    assert top.title == "Song B"


def test_top_candidate_empty_returns_none():
    assert Session().top_candidate() is None


# ── add_clue ───────────────────────────────────────────────────────────────────

def test_add_clue_appends():
    s = Session()
    s.add_clue("lyric", "something just like this")
    assert len(s.clues) == 1
    assert s.clues[0].type == "lyric"
    assert s.clues[0].value == "something just like this"
    assert s.clues[0].turn == 0


def test_add_clue_records_turn_at_time_of_addition():
    s = Session()
    s.add_message("user", "first message")   # turn increments to 1
    s.add_clue("context", "slow, melancholic")
    assert s.clues[0].turn == 1


# ── add_message / turn counting ───────────────────────────────────────────────

def test_turn_increments_on_user_message():
    s = Session()
    assert s.turn == 0
    s.add_message("user", "hello")
    assert s.turn == 1
    s.add_message("user", "more info")
    assert s.turn == 2


def test_turn_does_not_increment_on_assistant_message():
    s = Session()
    s.add_message("user", "hello")
    s.add_message("assistant", "response")
    assert s.turn == 1


def test_messages_stored_in_history():
    s = Session()
    s.add_message("user", "hello")
    s.add_message("assistant", "world")
    assert len(s.messages) == 2
    assert s.messages[0]["role"] == "user"
    assert s.messages[1]["role"] == "assistant"


# ── update_from_agent_response ─────────────────────────────────────────────────

def test_update_from_agent_response_replaces_candidates():
    s = Session()
    s.upsert_candidate("Old Song", "Old Artist", 0.2, "stale")
    response = {
        "updated_candidates": [
            {"title": "New Song A", "artist": "Artist A", "confidence": 0.9, "evidence": ["audio"]},
            {"title": "New Song B", "artist": "Artist B", "confidence": 0.4, "evidence": []},
        ]
    }
    s.update_from_agent_response(response)
    assert len(s.candidates) == 2
    titles = [c.title for c in s.candidates]
    assert "New Song A" in titles
    assert "Old Song" not in titles


def test_update_from_agent_response_sorts_descending():
    s = Session()
    response = {
        "updated_candidates": [
            {"title": "Low", "artist": "A", "confidence": 0.2, "evidence": []},
            {"title": "High", "artist": "B", "confidence": 0.9, "evidence": []},
            {"title": "Mid", "artist": "C", "confidence": 0.5, "evidence": []},
        ]
    }
    s.update_from_agent_response(response)
    assert s.candidates[0].title == "High"
    assert s.candidates[1].title == "Mid"
    assert s.candidates[2].title == "Low"


def test_update_from_agent_response_no_key_leaves_candidates_unchanged():
    s = Session()
    s.upsert_candidate("Existing", "Artist", 0.5, "old clue")
    s.update_from_agent_response({"message_to_user": "no candidates key here"})
    assert len(s.candidates) == 1
    assert s.candidates[0].title == "Existing"


def test_update_from_agent_response_propagates_optional_fields():
    s = Session()
    response = {
        "updated_candidates": [
            {
                "title": "Track X",
                "artist": "Artist X",
                "confidence": 0.75,
                "evidence": ["melody", "lyric"],
                "spotify_id": "abc123",
                "preview_url": "https://preview.example.com/track.mp3",
            }
        ]
    }
    s.update_from_agent_response(response)
    c = s.candidates[0]
    assert c.spotify_id == "abc123"
    assert c.preview_url == "https://preview.example.com/track.mp3"


# ── mark_resolved ──────────────────────────────────────────────────────────────

def test_mark_resolved():
    s = Session()
    s.upsert_candidate("Found It", "Artist", 0.95, "user confirmed")
    c = s.top_candidate()
    s.mark_resolved(c)
    assert s.resolved is True
    assert s.resolved_song is c
    assert s.resolved_song.title == "Found It"


# ── to_dict serialisation ──────────────────────────────────────────────────────

def test_to_dict_contains_required_keys():
    s = Session()
    d = s.to_dict()
    for key in ("session_id", "turn", "candidates", "clues_collected", "resolved"):
        assert key in d, f"Missing key: {key}"


def test_to_dict_candidates_serialised():
    s = Session()
    s.upsert_candidate("Song A", "Artist A", 0.8, "clue1")
    d = s.to_dict()
    assert len(d["candidates"]) == 1
    assert d["candidates"][0]["title"] == "Song A"
    assert d["candidates"][0]["confidence"] == 0.8


def test_to_dict_clues_serialised():
    s = Session()
    s.add_clue("lyric", "test lyric")
    d = s.to_dict()
    assert len(d["clues_collected"]) == 1
    assert d["clues_collected"][0]["type"] == "lyric"
    assert d["clues_collected"][0]["value"] == "test lyric"


def test_to_dict_session_id_is_string():
    s = Session()
    assert isinstance(s.to_dict()["session_id"], str)
    assert len(s.to_dict()["session_id"]) > 0
