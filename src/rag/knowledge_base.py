"""
Grounded Knowledge Synthesis Engine for MediNexus AI RAG.
Synthesizes retrieved healthcare policy and clinical guideline passages into cited answers.
"""

from typing import Dict, Any, Optional, List
import os
import requests

from config.settings import GEMINI_API_KEY, ENABLE_GEMINI
from src.rag.retriever import retrieve_context


def generate_grounded_answer(query: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve relevant knowledge chunks and synthesize a grounded response with source citations.
    """
    hits = retrieve_context(query, top_k=3, category=category)

    if not hits:
        return {
            "query": query,
            "answer": "No directly matching healthcare policies or clinical guidelines were found in the institutional knowledge base.",
            "citations": [],
            "grounded": False,
        }

    # Format context passages
    context_text = "\n\n---\n\n".join(
        [f"Source: [{h['title']} ({h['filename']})]\n{h['text']}" for h in hits]
    )

    citations = [
        {"title": h["title"], "filename": h["filename"], "category": h["category"], "relevance": h["score"]}
        for h in hits
    ]

    # Try Gemini API if key is available
    if ENABLE_GEMINI:
        try:
            prompt = (
                f"You are a clinical decision-support knowledge assistant. "
                f"Answer the user query strictly using the provided institutional healthcare policy documents. "
                f"Cite the source document title. "
                f"If the answer cannot be determined from the text, state that clearly.\n\n"
                f"DOCUMENTS:\n{context_text}\n\n"
                f"USER QUERY: {query}"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "query": query,
                    "answer": text,
                    "citations": citations,
                    "grounded": True,
                    "engine": "Gemini-1.5-Flash (Grounded RAG)",
                }
        except Exception:
            pass  # Fall through to deterministic local synthesis

    # Deterministic local knowledge synthesis
    top_hit = hits[0]
    local_synthesis = (
        f"According to institutional documentation in **{top_hit['title']}** (*{top_hit['filename']}*):\n\n"
        f"> {top_hit['text'][:350]}...\n\n"
        f"**Key Operational Takeaways:**\n"
        f"- Reference Guideline ID: `{top_hit['filename']}`\n"
        f"- Target Department: `{top_hit['category']}`\n"
        f"- Relevance Match Score: `{top_hit['score'] * 100:.1f}%`"
    )

    return {
        "query": query,
        "answer": local_synthesis,
        "citations": citations,
        "grounded": True,
        "engine": "MediNexus Local Grounded Retriever",
    }
