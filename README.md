# whats-that-tune

You know that feeling - a song is stuck in your head but you can't remember what it's called. You hum it to people and they shrug.

This is a fix for that.

Hum into your mic. Describe how it felt. Say a word you think was in it. The agent asks you one question at a time and narrows it down until it finds it.

Works for Western and Indian film music.

## how it works

One question at a time. Each answer rules out more songs until only one is left. Full diagram in [docs/how-it-works.md](docs/how-it-works.md).

## works best when you

Type a lyric, even two or three words.

Say the language or film name if you know it.

Hum first, then describe.

## run it

**Web UI**
```
pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --reload
```
Open `http://localhost:8000`.

**Terminal**
```
python cli.py
python cli.py --audio files/my_hum.wav
```

**Tests**
```
pytest tests/
```

## api keys

`ANTHROPIC_API_KEY` is required. Everything else is optional but improves results.

- `GENIUS_API_TOKEN` - lyrics search. Get one at genius.com/api-clients.
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` - Western music context and previews.
- JioSaavn works without any key.

## melody matching

Melody search is disabled until you build a corpus index from your own audio files. Add audio files and run:

```
python -c "
from matcher import ChromaMatcher
m = ChromaMatcher()
m.add_directory('your_music_folder', metadata={})
m.save('corpus/chroma_index.pkl')
"
```

Everything else works without it.

## stack

Python · FastAPI · Claude (Anthropic) · librosa · Genius API · JioSaavn · Spotify API
