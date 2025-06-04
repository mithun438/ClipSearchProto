# ClipSearch — Semantic Video Search Engine (Offline)

ClipSearch is a lightweight, offline tool that allows users to **search through video transcripts** and **jump directly to the relevant timestamp** in a YouTube video. It uses **local AI models** for semantic search — no OpenAI API or internet needed (after transcript is fetched).

---

##Features

- Search video content by **meaning**, not just keywords
- Uses `sentence-transformers` for **local embeddings**
- Fast semantic search via `FAISS`
- Jump directly to exact timestamp in the video
- Works offline after downloading transcript

---

##Tech Stack

| Tool | Purpose |
|------|---------|
| `youtube-transcript-api` | Fetch video subtitles (YouTube only) |
| `sentence-transformers` | Convert transcript chunks into embeddings |
| `FAISS` | Vector-based semantic search engine |
| `NumPy` | Numerical operations |
| `Python` | Main language |

---
