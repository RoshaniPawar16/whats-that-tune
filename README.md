# whats-that-tune

You know that feeling - a song is stuck in your head but you can't remember what it's called. You hum it to people and they shrug.

This is a fix for that.

Hum into your mic. Describe how it felt. Say a word you think was in it. The agent asks you one question at a time and narrows it down until it finds it.

Works for Western and Indian film music.

## how it works

The agent maintains a belief state - a ranked list of candidate songs with confidence scores - and narrows it down across turns using audio analysis, lyrics search, and context clues. Full diagram in [docs/how-it-works.md](docs/how-it-works.md).

## works best when you

Type a lyric fragment, even two or three words - lyrics are the fastest path to a match. Say the language or film name if you know it - Bollywood, Tamil, Telugu, or the film title will narrow it down immediately. Hum first and then add context - the audio gives Mnemo a melodic fingerprint, and your description rules out the rest.

## run it

```
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY, then optionally GENIUS_API_TOKEN and Spotify credentials
uvicorn api:app --reload
# open http://localhost:8000
```

## stack

Python · FastAPI · Claude (Anthropic) · librosa · Genius API · JioSaavn · Spotify API
