# How Mnemo Works

```mermaid
flowchart TD
    A([User sends a clue\ntext · audio · or both]) --> B[Agent reads clue\nand conversation history]

    B --> C{Which tools\nto call?}

    C -->|Audio present| D[analyse_audio\nlibrosa pyin\nextracts pitch · tempo · key]
    C -->|Lyric fragment| E[search_lyrics\nGenius API]
    C -->|Indian / Bollywood / regional| F[search_jiosaavn\njiosaavn.com · no auth]
    C -->|Western context / decade / genre| G[search_context\nSpotify search API]
    C -->|Pitch data + corpus exists| H[search_melody\nchromagram DTW\nneeds corpus index]

    D --> I[Agent ranks candidates\nby confidence]
    E --> I
    F --> I
    G --> I
    H --> I

    I --> J{Top candidate\nconfidence > 0.7?}

    J -->|No| K[Ask ONE discriminative question]
    K --> A

    J -->|Yes| L[play_candidate\nSpotify preview or JioSaavn perma_url]
    L --> M{User confirms?}
    M -->|No - wrong song| K
    M -->|Yes| N([Song found])
```

The agent (Claude) decides which tools to call each turn based on what the user said. There is no fixed routing - it reasons over the clues and picks the most useful search.

**Working now:** audio analysis, Genius lyrics search, JioSaavn Indian song search, Spotify context search.

**Needs setup:** Spotify preview playback needs `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env`. Melody search needs a local corpus index built first - see README.
