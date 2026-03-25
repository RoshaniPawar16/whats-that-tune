# How Mnemo Works

```mermaid
flowchart TD
    A([User hums, sings, or describes]) --> B{Input type?}

    B -->|Audio| C[Audio Analysis\nlibrosa pyin]
    B -->|Lyrics / text| E2
    B -->|Indian / Bollywood context| E3
    B -->|Western context| G

    C --> D[Feature Extraction\npitch sequence · tempo · key · chroma]

    D --> E1[Chromagram DTW\nmelody matching\nneeds corpus index]
    D --> E2

    E2[Lyrics Search\nGenius API] --> F
    E3[JioSaavn Search\njiosaavn.com · no auth] --> F
    E1 --> F

    G[Context Search\nSpotify · decade · genre · mood\nWestern music] --> F

    F[Score Fusion\nweighted candidate ranking] --> H

    H[Belief State Update\ncandidate list · confidence scores · evidence] --> I

    I{Confident?\ntop score > 0.7}

    I -->|No| J[Agent picks ONE\ndiscriminative question]
    J --> A

    I -->|Yes| K[Play Preview\nSpotify or JioSaavn perma_url]
    K --> L{User confirms?}
    L -->|No - wrong song| J
    L -->|Yes| M([Song found])
```

**What's wired:** Genius lyrics search, JioSaavn Indian song search, Spotify context search, audio analysis via librosa.

**Still needs setup:** Chromagram DTW melody matching requires a corpus index — see README for build instructions. Spotify preview playback requires `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env`.
