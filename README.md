# 🫙 Pensieve

**Pour in your content. Explore it at will.**

A local-first NotebookLM-lite app built with Streamlit. Upload PDFs, text files, or paste YouTube URLs — get auto-generated summaries and chat with your sources using RAG (Retrieval-Augmented Generation).

Named after Dumbledore's Pensieve — a magical basin where you pour in memories and explore them later. Here you pour in content and explore it through conversation.

---

## Features

- **📄 Multi-format ingestion** — PDF, TXT files, and YouTube video transcripts
- **🤖 Auto-summaries** — Every uploaded source gets an instant AI summary
- **💬 Chat with your sources** — Ask questions and get answers grounded in your actual content
- **🎙️ Podcast generation** — Two AI hosts discuss your sources in a generated audio podcast
- **🗑️ Source management** — Delete sources you no longer need
- **💾 Persistent storage** — ChromaDB keeps your data across app restarts
- **🔒 Local-first** — Everything runs on your machine, only API calls go to Gemini

## Architecture

```
[Upload: PDF / TXT / YouTube URL]
         ↓
[Document Loader] — PyPDF2 / youtube-transcript-api / open()
         ↓
[Text Chunker] — RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
         ↓
[Embeddings] — Gemini gemini-embedding-001 (3072 dims)
         ↓
[Vector Store] — ChromaDB (embedded, persisted to ./chroma_db/)
         ↓
[User Question] → [Embed → Top-5 Retrieval] → [Gemini + Context] → [Grounded Answer]
```

## Tech Stack

| Component        | Technology                              |
|------------------|-----------------------------------------|
| Frontend         | Streamlit                               |
| LLM              | Google Gemini 2.5 Flash (free tier)     |
| Embeddings       | Gemini gemini-embedding-001 (3072-dim)  |
| Vector Store     | ChromaDB (embedded, local)              |
| PDF Parsing      | PyPDF2                                  |
| YouTube          | youtube-transcript-api                  |
| Text Splitting   | langchain-text-splitters                || Podcast TTS      | edge-tts (Microsoft neural voices)      |
## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/InduSridhar/pensieve.git
cd pensieve
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your API key

Get a **free** API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# Edit .env and paste your key
```

### 4. Run the app

```bash
streamlit run app.py
```

## Project Structure

```
pensieve/
├── .env.example              # GOOGLE_API_KEY=your-key-here
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                    # Streamlit entry point
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Env vars, model names, chunk settings
│   ├── document_loader.py    # PDF, TXT, YouTube transcript loading
│   ├── chunker.py            # Text splitting with metadata
│   ├── embeddings.py         # Gemini embedding wrapper
│   ├── vector_store.py       # ChromaDB operations (add, query, delete)
│   ├── rag_pipeline.py       # Orchestrator: ingest, query, summarize
│   ├── chat.py               # Prompt templates, conversation management
│   └── podcast.py            # Podcast script generation + TTS + audio stitching
│
├── chroma_db/                # Auto-created, gitignored
├── uploads/                  # Temp uploaded files, gitignored
│
└── tests/
    ├── test_chunker.py
    ├── test_loader.py
    └── test_rag.py
```

## Running Tests

```bash
pytest tests/ -v
```

## How It Works

1. **Ingest**: Document → parsed to raw text → split into ~1000 char overlapping chunks → each chunk gets metadata
2. **Embed**: Each chunk → Gemini Embedding API → 3072-dimensional vector
3. **Store**: Vectors + chunk text + metadata → saved in ChromaDB (persists on disk)
4. **Query**: User question → embedded → cosine similarity search → top 5 most relevant chunks retrieved
5. **Generate**: Chunks + question → structured prompt → Gemini → grounded answer citing sources
6. **Summarize**: On upload, first N chunks sent to Gemini with a "summarize" prompt → auto-summary displayed

## License

MIT
