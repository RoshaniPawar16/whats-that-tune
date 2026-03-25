# whats-that-tune

You know that feeling - a song is stuck in your head but you can't remember what it's called. You hum it to people and they shrug.

This is a fix for that.

Hum into your mic. Describe how it felt. Say a word you think was in it. The agent asks you one question at a time and narrows it down until it finds it.

Works for Western and Indian film music.

## how it works

[how it works &rarr;](docs/how-it-works.md)

## run it

```
pip install -r requirements.txt
cp .env.example .env
python cli.py
```

## stack

Python · FastAPI · Claude (Anthropic) · librosa · Spotify API
