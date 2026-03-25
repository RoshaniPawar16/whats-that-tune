"""
ChromaMatcher: chromagram-based song corpus index with DTW query.

Build a corpus:
    matcher = ChromaMatcher()
    matcher.add_file("song.wav", title="Bohemian Rhapsody", artist="Queen")
    matcher.save("corpus/chroma_index.pkl")

Query:
    matcher = ChromaMatcher.load("corpus/chroma_index.pkl")
    results = matcher.query(["C4", "E4", "G4"], top_k=5)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np


@dataclass
class CorpusEntry:
    """One song in the corpus."""
    title: str
    artist: str
    chroma: np.ndarray   # shape (12, T) — CQT chroma
    notes: list[str]     # deduplicated pitch sequence extracted at index time
    path: str            # original audio path (informational only)


# Chroma pitch-class index (handles enharmonic equivalents)
_NOTE_TO_IDX: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


class ChromaMatcher:
    """Chromagram corpus index with DTW-based melodic similarity search."""

    def __init__(self) -> None:
        self.corpus: list[CorpusEntry] = []

    # ── Index building ─────────────────────────────────────────────────────────

    def add_file(self, path: str, title: str, artist: str) -> None:
        """Extract chroma + pitch sequence from an audio file and add to corpus."""
        y, sr = librosa.load(path, sr=16000, mono=True)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
        notes: list[str] = []
        for freq, voiced in zip(f0, voiced_flag):
            if voiced and freq and not np.isnan(freq):
                note = librosa.hz_to_note(freq)
                if not notes or notes[-1] != note:
                    notes.append(note)

        self.corpus.append(
            CorpusEntry(title=title, artist=artist, chroma=chroma, notes=notes, path=path)
        )

    def add_directory(self, directory: str, metadata: dict[str, dict] | None = None) -> None:
        """Index all .wav and .mp3 files in a directory.

        Args:
            directory: Path to scan for audio files.
            metadata: Optional dict mapping filename stem → {"title": ..., "artist": ...}.
                      Filename stem is used as title if not provided.
        """
        audio_dir = Path(directory)
        for audio_path in sorted(audio_dir.glob("**/*")):
            if audio_path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac"}:
                continue
            stem = audio_path.stem
            info = (metadata or {}).get(stem, {})
            title = info.get("title", stem)
            artist = info.get("artist", "Unknown")
            self.add_file(str(audio_path), title=title, artist=artist)

    def save(self, path: str) -> None:
        """Persist the corpus index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.corpus, f)

    @classmethod
    def load(cls, path: str) -> "ChromaMatcher":
        """Load a previously saved corpus index."""
        matcher = cls()
        with open(path, "rb") as f:
            matcher.corpus = pickle.load(f)
        return matcher

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, pitch_sequence: list[str], top_k: int = 5) -> list[dict]:
        """Return the top_k corpus entries most similar to the query pitch sequence.

        Args:
            pitch_sequence: Note names, e.g. ["C4", "E4", "G4"]. Use "X" for unknowns.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with title, artist, confidence, source keys, sorted by confidence.
        """
        if not self.corpus:
            return []

        query_chroma = self._notes_to_chroma(pitch_sequence)

        scored: list[tuple[float, CorpusEntry]] = []
        for entry in self.corpus:
            dist = self._dtw_distance(query_chroma, entry.chroma)
            scored.append((dist, entry))

        scored.sort(key=lambda x: x[0])

        return [
            {
                "title": entry.title,
                "artist": entry.artist,
                "confidence": self._dist_to_confidence(dist),
                "source": "melody_dtw",
            }
            for dist, entry in scored[:top_k]
        ]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _notes_to_chroma(self, notes: list[str]) -> np.ndarray:
        """Convert a pitch sequence to a (12, len(notes)) chroma indicator matrix."""
        n = max(len(notes), 1)
        chroma = np.zeros((12, n), dtype=float)
        for i, note in enumerate(notes):
            if note == "X":
                continue
            # Strip octave digit(s): "C#4" → "C#"
            pitch_class = "".join(c for c in note if not c.isdigit())
            idx = _NOTE_TO_IDX.get(pitch_class)
            if idx is not None:
                chroma[idx, i] = 1.0
        return chroma

    def _dtw_distance(self, query: np.ndarray, reference: np.ndarray) -> float:
        """Compute DTW accumulated cost between two chroma matrices."""
        D, _ = librosa.sequence.dtw(X=query, Y=reference, metric="cosine")
        return float(D[-1, -1])

    def _dist_to_confidence(self, dist: float) -> float:
        """Map DTW distance → [0.05, 0.90] confidence.

        Calibration note: the mapping dist=0 → 0.90, dist→∞ → 0.05 is a soft
        exponential decay. Re-calibrate the scale factor (5.0) against your corpus
        once you have representative query/match pairs.
        """
        return float(np.clip(0.9 * np.exp(-dist / 5.0), 0.05, 0.90))
