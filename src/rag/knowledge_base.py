"""
Grounded Knowledge Synthesis Engine for MediNexus AI RAG.
Synthesizes retrieved healthcare policy and clinical guideline passages into cited answers.
Supports Google Gemini, OpenAI, xAI Grok, and high-precision deterministic local synthesis.
"""

from typing import Dict, Any, Optional, List
import os
import time
import requests
import json

from config.settings import get_active_llm_config
from src.rag.retriever import retrieve_context


def _clean_passage_text(raw_text: str) -> str:
    """Strip synthetic demo headers and metadata prefixes from document text."""
    lines = raw_text.splitlines()
    clean = []
    for line in lines:
        s = line.strip()
        if not s or any(s.startswith(p) for p in ["[DEMO", "DOCUMENT TITLE:", "DOCUMENT ID:", "REVISION:", "EFFECTIVE DATE:"]):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def generate_grounded_answer(query: str, category: Optional[str] = None, min_confidence: float = 0.12) -> Dict[str, Any]:
    """
    Retrieve relevant knowledge chunks and synthesize a grounded response with source citations.
    Supports Google Gemini, OpenAI (gpt-4o-mini), xAI Grok, and deterministic local synthesis.
    Enforces a clinical safety confidence threshold (min_confidence) to safely reject out-of-scope queries.
    Records high-precision latency telemetry (retrieval_ms, generation_ms, total_ms).
    """
    t_start = time.perf_counter()
    hits = retrieve_context(query, top_k=3, category=category)
    t_ret = time.perf_counter()

    if not hits or hits[0]["score"] < min_confidence:
        t_end = time.perf_counter()
        return {
            "query": query,
            "answer": (
                "No authoritative institutional healthcare policies or clinical guidelines were found in the "
                "MediNexus knowledge base for this query with sufficient clinical confidence. To protect patient "
                "safety and prevent clinical hallucinations, MediNexus AI safely refuses to answer unsupported queries."
            ),
            "citations": [],
            "grounded": False,
            "out_of_scope": True,
            "safe_refusal": True,
            "engine": "Clinical Safety Guardrail",
            "latency_metrics": {
                "retrieval_ms": round((t_ret - t_start) * 1000, 2),
                "generation_ms": round((t_end - t_ret) * 1000, 2),
                "total_ms": round((t_end - t_start) * 1000, 2),
            },
            "top_relevance_score": hits[0]["score"] if hits else 0.0,
        }

    # Format context passages
    context_text = "\n\n---\n\n".join(
        [f"Source: [{h['title']} ({h['filename']})]\n{h['text']}" for h in hits]
    )

    citations = [
        {"title": h["title"], "filename": h["filename"], "category": h["category"], "relevance": h["score"]}
        for h in hits
    ]

    rag_prompt = (
        f"You are a clinical decision-support knowledge assistant for MediNexus AI. "
        f"Answer the user query strictly and factually using the provided institutional healthcare policy documents below. "
        f"Always cite the source document name and specific clinical criteria (such as timelines, dosages, or thresholds). "
        f"If the answer cannot be determined directly from the text, state that clearly.\n\n"
        f"INSTITUTIONAL DOCUMENTS:\n{context_text}\n\n"
        f"USER CLINICAL QUERY: {query}"
    )

    llm_config = get_active_llm_config()

    # 1. Google Gemini API
    if llm_config["provider"] == "gemini" and llm_config["api_key"]:
        try:
            api_key = llm_config["api_key"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": rag_prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
            }
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                t_end = time.perf_counter()
                return {
                    "query": query,
                    "answer": text,
                    "citations": citations,
                    "grounded": True,
                    "engine": "Gemini 1.5 Flash (Grounded RAG)",
                    "latency_metrics": {
                        "retrieval_ms": round((t_ret - t_start) * 1000, 2),
                        "generation_ms": round((t_end - t_ret) * 1000, 2),
                        "total_ms": round((t_end - t_start) * 1000, 2),
                    },
                    "top_relevance_score": hits[0]["score"] if hits else 0.0,
                }
        except Exception:
            pass  # Fall through to local fallback

    # 2. OpenAI API
    if llm_config["provider"] == "openai" and llm_config["api_key"]:
        try:
            api_key = llm_config["api_key"]
            model_name = llm_config.get("model", "gpt-4o-mini")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a clinical decision-support knowledge assistant."},
                    {"role": "user", "content": rag_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 800,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                t_end = time.perf_counter()
                return {
                    "query": query,
                    "answer": text,
                    "citations": citations,
                    "grounded": True,
                    "engine": f"OpenAI ({model_name}) Grounded RAG",
                    "latency_metrics": {
                        "retrieval_ms": round((t_ret - t_start) * 1000, 2),
                        "generation_ms": round((t_end - t_ret) * 1000, 2),
                        "total_ms": round((t_end - t_start) * 1000, 2),
                    },
                    "top_relevance_score": hits[0]["score"] if hits else 0.0,
                }
        except Exception:
            pass  # Fall through to local fallback

    # 3. xAI Grok API
    if llm_config["provider"] == "grok" and llm_config["api_key"]:
        try:
            api_key = llm_config["api_key"]
            model_name = llm_config.get("model", "grok-beta")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a clinical decision-support knowledge assistant."},
                    {"role": "user", "content": rag_prompt},
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                t_end = time.perf_counter()
                return {
                    "query": query,
                    "answer": text,
                    "citations": citations,
                    "grounded": True,
                    "engine": f"Grok ({model_name}) Grounded RAG",
                    "latency_metrics": {
                        "retrieval_ms": round((t_ret - t_start) * 1000, 2),
                        "generation_ms": round((t_end - t_ret) * 1000, 2),
                        "total_ms": round((t_end - t_start) * 1000, 2),
                    },
                    "top_relevance_score": hits[0]["score"] if hits else 0.0,
                }
        except Exception:
            pass  # Fall through to local fallback

    # 4. Deterministic Local Knowledge Synthesis
    top_hit = hits[0]
    cleaned_guidance = _clean_passage_text(top_hit["text"])

    local_synthesis = (
        f"According to institutional documentation in **{top_hit['title']}** (*{top_hit['filename']}*):\n\n"
        f"{cleaned_guidance}\n\n"
        f"**Key Operational Takeaways:**\n"
        f"- Reference Guideline ID: `{top_hit['filename']}`\n"
        f"- Target Department: `{top_hit['category']}`\n"
        f"- Relevance Match Score: `{top_hit['score'] * 100:.1f}%`"
    )

    t_end = time.perf_counter()
    return {
        "query": query,
        "answer": local_synthesis,
        "citations": citations,
        "grounded": True,
        "engine": "MediNexus Local Grounded Retriever",
        "latency_metrics": {
            "retrieval_ms": round((t_ret - t_start) * 1000, 2),
            "generation_ms": round((t_end - t_ret) * 1000, 2),
            "total_ms": round((t_end - t_start) * 1000, 2),
        },
        "top_relevance_score": hits[0]["score"] if hits else 0.0,
    }
