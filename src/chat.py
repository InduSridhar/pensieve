"""
Chat & Prompt Templates for Pensieve.
Handles prompt construction for RAG queries, summarisation, and conversation management.
"""

from __future__ import annotations

from typing import Any


# ── System Prompts ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Pensieve, a helpful research assistant. You answer questions based ONLY on the provided context from the user's uploaded documents.

Rules:
1. Answer based on the context provided below. Be thorough and specific.
2. If the answer is NOT in the context, say: "I couldn't find information about that in your uploaded sources."
3. When possible, mention which source the information comes from.
4. Be concise but complete. Use bullet points for lists.
5. Do not make up information that isn't in the context."""

SUMMARY_PROMPT_TEMPLATE = """You are a document summarizer. Provide a clear, well-structured summary of the following content.

Source: {source_name}

Content:
{content}

Instructions:
- Write a concise summary (3-5 paragraphs)
- Highlight the key points, themes, and takeaways
- Use bullet points for important details
- Keep it informative and well-organized"""


# ── Prompt Builders ──────────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    context_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Build a full RAG prompt with system instructions, context, history, and question.

    Args:
        question: The user's current question.
        context_chunks: Retrieved chunks with 'text' and 'source' keys.
        chat_history: Optional list of previous messages.

    Returns:
        A formatted prompt string ready for Gemini.
    """
    # Build context section
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {source}]\n{text}")

    context_section = "\n\n---\n\n".join(context_parts)

    # Build conversation history section
    history_section = ""
    if chat_history:
        history_lines = []
        # Include last 6 messages for context (3 exchanges)
        recent_history = chat_history[-6:]
        for msg in recent_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_section = "\n\nPrevious conversation:\n" + "\n".join(history_lines)

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT FROM UPLOADED SOURCES ---

{context_section}

--- END CONTEXT ---
{history_section}

User question: {question}

Answer:"""

    return prompt


def build_summary_prompt(content: str, source_name: str) -> str:
    """
    Build a summarisation prompt.

    Args:
        content: The text to summarize.
        source_name: Name of the source document.

    Returns:
        A formatted summary prompt string.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        source_name=source_name,
        content=content,
    )
