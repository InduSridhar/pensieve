"""
Pensieve — Streamlit Entry Point
Pour in your content. Explore it at will.
"""

import streamlit as st

from src.config import validate_api_key, UPLOADS_DIR
from src import vector_store
from src.rag_pipeline import ingest_pdf, ingest_txt, ingest_youtube, query
from src.podcast import generate_podcast

import os

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pensieve",
    page_icon="🫙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialisation ─────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "summaries" not in st.session_state:
    st.session_state.summaries = {}

if "podcast_audio" not in st.session_state:
    st.session_state.podcast_audio = None

if "podcast_script" not in st.session_state:
    st.session_state.podcast_script = None


# ── Helper Functions ─────────────────────────────────────────────────

def refresh_sources():
    """Refresh the source list from the vector store."""
    st.session_state.sources = vector_store.list_sources()


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to the uploads directory and return the path."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ── API Key Check ────────────────────────────────────────────────────

if not validate_api_key():
    st.error(
        "⚠️ **Google API Key not configured.**\n\n"
        "1. Get a free key from [Google AI Studio](https://aistudio.google.com/apikey)\n"
        "2. Copy `.env.example` to `.env` and paste your key\n"
        "3. Restart the app"
    )
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🫙 Pensieve")
    st.caption("Pour in your content. Explore it at will.")

    st.divider()

    # ── File Upload ──
    st.subheader("📄 Upload Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT",
    )

    if uploaded_file and st.button("📥 Ingest File", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                save_uploaded_file(uploaded_file)

                if uploaded_file.name.lower().endswith(".pdf"):
                    result = ingest_pdf(uploaded_file, uploaded_file.name)
                else:
                    result = ingest_txt(uploaded_file, uploaded_file.name)

                st.session_state.summaries[result["source"]] = result["summary"]
                refresh_sources()
                st.success(f"✅ Ingested **{result['source']}** ({result['num_chunks']} chunks)")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()

    # ── YouTube URL ──
    st.subheader("🎬 YouTube")

    youtube_url = st.text_input(
        "Paste a YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if youtube_url and st.button("📥 Ingest YouTube", use_container_width=True):
        with st.spinner("Fetching transcript and processing..."):
            try:
                result = ingest_youtube(youtube_url)
                st.session_state.summaries[result["source"]] = result["summary"]
                refresh_sources()
                st.success(f"✅ Ingested YouTube video ({result['num_chunks']} chunks)")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()

    # ── Source Management ──
    st.subheader("📚 Your Sources")

    refresh_sources()
    sources = st.session_state.sources

    if not sources:
        st.info("No sources yet. Upload a document or paste a YouTube URL to get started.")
    else:
        for source_info in sources:
            source_name = source_info["source"]
            source_type = source_info["type"]
            num_chunks = source_info["num_chunks"]

            type_emoji = {"pdf": "📄", "txt": "📝", "youtube": "🎬"}.get(source_type, "📎")

            col1, col2 = st.columns([4, 1])
            with col1:
                # Truncate long names
                display_name = source_name if len(source_name) <= 30 else source_name[:27] + "..."
                st.markdown(f"{type_emoji} **{display_name}**  \n_{num_chunks} chunks_")
            with col2:
                if st.button("🗑️", key=f"del_{source_name}", help=f"Delete {source_name}"):
                    deleted = vector_store.delete_source(source_name)
                    st.session_state.summaries.pop(source_name, None)
                    refresh_sources()
                    st.toast(f"Deleted {source_name} ({deleted} chunks)")
                    st.rerun()

    st.divider()

    # ── Podcast Generation ──
    st.subheader("🎙️ Podcast")

    if not sources:
        st.caption("Upload sources to generate a podcast.")
    else:
        if st.button("🎙️ Generate Podcast", use_container_width=True):
            progress_bar = st.progress(0, text="Starting...")

            def update_progress(step, total, msg):
                progress_bar.progress(step / total, text=msg)

            try:
                audio, script = generate_podcast(
                    progress_callback=update_progress,
                )
                st.session_state.podcast_audio = audio
                st.session_state.podcast_script = script
                progress_bar.empty()
                st.success("✅ Podcast ready!")
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Error: {e}")

    st.divider()

    # ── Clear Chat ──
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # ── Stats ──
    total_chunks = vector_store.count()
    st.caption(f"Total chunks in store: {total_chunks}")


# ── Main Area ────────────────────────────────────────────────────────

st.title("🫙 Pensieve")
st.caption("Upload documents and YouTube videos, then chat with your sources.")

# ── Show Summaries ──
if st.session_state.summaries:
    with st.expander("📋 Source Summaries", expanded=False):
        for source_name, summary in st.session_state.summaries.items():
            st.markdown(f"### {source_name}")
            st.markdown(summary)
            st.divider()

# ── Podcast Player ──
if st.session_state.podcast_audio:
    with st.expander("🎙️ Podcast", expanded=True):
        st.audio(st.session_state.podcast_audio, format="audio/mp3")

        # Show script
        if st.session_state.podcast_script:
            st.markdown("**Script:**")
            for entry in st.session_state.podcast_script:
                speaker = entry["speaker"]
                line = entry["line"]
                icon = "🟦" if speaker == "Ava" else "🟧"
                st.markdown(f"{icon} **{speaker}:** {line}")

# ── Chat History ──
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ──
if not sources:
    st.info("👆 Upload a document or paste a YouTube URL in the sidebar to start chatting.")
else:
    if user_input := st.chat_input("Ask a question about your sources..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query(user_input, st.session_state.chat_history[:-1])

                    st.markdown(result["answer"])

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
