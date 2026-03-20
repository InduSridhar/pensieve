"""
Podcast Generator for Pensieve.
Generates a two-host podcast-style audio discussion from source content.

Pipeline: source chunks → script (Gemini) → TTS per line (edge-tts) → stitched MP3
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Any

import edge_tts
from google import genai

from src.config import GOOGLE_API_KEY, CHAT_MODEL
from src import vector_store


# ── Constants ────────────────────────────────────────────────────────

VOICE_HOST_A = "en-US-AvaNeural"       # Female — "Expressive, Caring, Pleasant"
VOICE_HOST_B = "en-US-AndrewNeural"   # Male — "Warm, Confident, Authentic"

PODCAST_SCRIPT_PROMPT = """You are a top-tier podcast script writer for a show called "Deep Dive". Write a lively, engaging conversation between two hosts who have both thoroughly studied the source material.

HOST PERSONALITIES:
- Ava: Sharp, opinionated, articulate. Has strong takes and backs them up. Notices patterns and connections others miss. Speaks with authority but is open to being challenged.
- Andrew: Equally knowledgeable but from a different angle. Plays devil's advocate. Brings up counterpoints, alternative perspectives, and real-world implications. Not a student — a peer who sometimes disagrees.

BOTH hosts are educated, well-read, and opinionated. Neither is "teaching" the other. They are two smart people having a genuine discussion where they sometimes agree, sometimes push back, and both contribute original insights.

CRITICAL CONTENT RULES:
- Your script MUST directly reference and discuss the specific people, events, claims, and details from the source content
- If the source is a conversation or interview, reference what the speakers actually said ("When they mentioned X..." or "Their point about Y was interesting because...")
- Use specific names, numbers, quotes, and details from the source — do NOT be vague or generic
- At least 70% of the dialogue should be about concrete content from the source, not general commentary

STYLE GUIDE:
- Sound like two peers debating at a dinner party, NOT a teacher-student dynamic
- Both hosts share opinions: "I actually think...", "I disagree, here's why...", "What struck me was..."
- Include genuine disagreement or different interpretations at least once
- Use natural speech: "So basically...", "Right, but here's the thing...", "Hmm, I see it differently..."
- End with each host sharing their key takeaway (which should differ)

RULES:
1. Write 10-14 exchanges (20-28 total lines)
2. STRUCTURE the conversation in three acts:
   - ACT 1 (first 3-4 exchanges): Set the scene. What is this source about? Who's involved? Give the listener a quick recap of the big picture so they have context.
   - ACT 2 (next 5-7 exchanges): Go deeper into 2-3 key topics or claims. Discuss, debate, react to specifics.
   - ACT 3 (last 2-3 exchanges): Zoom out. What's the bigger takeaway? Each host shares a different final thought.
3. Each line is 1-3 sentences. Short lines create rhythm.
4. Do NOT use stage directions, emotes, asterisks, or descriptions — dialogue ONLY
5. Do NOT use markdown formatting of any kind

FORMAT — use EXACTLY this format, one line per speaker:
Ava: [dialogue]
Andrew: [dialogue]

SOURCE CONTENT:
{content}"""


# ── Script Generation ────────────────────────────────────────────────

def generate_script(source_name: str | None = None) -> list[dict[str, str]]:
    """
    Generate a podcast script from source content.

    Args:
        source_name: Specific source to use, or None for all sources.

    Returns:
        List of dicts with 'speaker' ('Ava' or 'Andrew') and 'line'.
    """
    # Gather content
    if source_name:
        chunks = vector_store.get_source_chunks(source_name)
    else:
        # Use all sources
        sources = vector_store.list_sources()
        chunks = []
        for src in sources:
            chunks.extend(vector_store.get_source_chunks(src["source"]))

    if not chunks:
        raise ValueError("No content available to generate a podcast.")

    # Combine chunks up to ~10000 chars
    combined = ""
    for chunk in chunks:
        if len(combined) + len(chunk["text"]) > 10000:
            break
        combined += chunk["text"] + "\n\n"

    # Generate script via Gemini
    prompt = PODCAST_SCRIPT_PROMPT.format(content=combined)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    # Parse the script into structured lines
    return _parse_script(response.text)


def _parse_script(raw_script: str) -> list[dict[str, str]]:
    """Parse the raw script text into speaker/line pairs."""
    lines = []
    for line in raw_script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match "Ava: ..." or "Andrew: ..."
        match = re.match(r"^(Ava|Andrew):\s*(.+)$", line, re.IGNORECASE)
        if match:
            speaker = match.group(1).capitalize()
            dialogue = match.group(2).strip()
            if dialogue:
                lines.append({"speaker": speaker, "line": dialogue})

    if not lines:
        raise ValueError("Failed to parse podcast script — no valid lines found.")

    return lines


# ── Text-to-Speech (edge-tts) ────────────────────────────────────────

async def _synthesize_line_async(text: str, voice: str) -> bytes:
    """
    Synthesize a single line of dialogue to MP3 audio using edge-tts.

    Args:
        text: The dialogue text.
        voice: The edge-tts voice name.

    Returns:
        MP3 audio bytes.
    """
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


def _synthesize_line(text: str, voice: str) -> bytes:
    """Sync wrapper for async TTS synthesis."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context (e.g. Streamlit),
            # create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _synthesize_line_async(text, voice))
                return future.result()
        else:
            return loop.run_until_complete(_synthesize_line_async(text, voice))
    except RuntimeError:
        return asyncio.run(_synthesize_line_async(text, voice))


# ── Main Entry Point ─────────────────────────────────────────────────

def generate_podcast(
    source_name: str | None = None,
    progress_callback: Any = None,
) -> tuple[bytes, list[dict[str, str]]]:
    """
    Generate a full podcast episode from source content.

    Args:
        source_name: Specific source, or None for all.
        progress_callback: Optional callable(step, total, message) for progress updates.

    Returns:
        Tuple of (MP3 audio bytes, script lines).
    """
    def _progress(step: int, total: int, msg: str):
        if progress_callback:
            progress_callback(step, total, msg)

    # Step 1: Generate script
    _progress(0, 1, "Writing podcast script...")
    script = generate_script(source_name)

    total_steps = len(script) + 1  # +1 for script generation
    _progress(1, total_steps, "Script ready! Generating audio...")

    # Step 2: Synthesize each line
    audio_clips: list[bytes] = []

    for i, entry in enumerate(script):
        speaker = entry["speaker"]
        line = entry["line"]
        voice = VOICE_HOST_A if speaker == "Ava" else VOICE_HOST_B

        _progress(i + 2, total_steps, f"🎙️ {speaker}: \"{line[:50]}...\"")

        audio = _synthesize_line(line, voice)
        audio_clips.append(audio)

    # Step 3: Concatenate MP3 clips (MP3 frames can be concatenated directly)
    _progress(total_steps, total_steps, "Stitching audio...")
    combined_audio = b"".join(audio_clips)

    return combined_audio, script
