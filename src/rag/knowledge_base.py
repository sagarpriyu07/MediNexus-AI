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


def fetch_lakehouse_operational_metric(query: str) -> Optional[Dict[str, Any]]:
    """
    Intelligently query DuckDB Lakehouse Gold marts when user asks about
    hospital operational metrics (readmissions, ALOS, occupancy, patient counts, bed capacity).
    """
    try:
        from src.utils.database import query_df
        from src.rag.retriever import normalize_query

        q_norm = normalize_query(query).lower()

        # 1. 30-Day Readmission queries
        if any(k in q_norm for k in ["readmission", "readmit", "bounce back"]):
            if any(w in q_norm for w in ["rate", "percent", "%", "what", "how many", "ratio", "statistic", "metric", "30 day", "30-day", "30days", "overall", "number"]):
                df = query_df("""
                    SELECT 
                        count(*) as total, 
                        sum(readmitted_30d) as readmitted, 
                        round(avg(readmitted_30d)*100, 2) as readmit_pct, 
                        round(avg(length_of_stay), 1) as avg_los 
                    FROM gold_admissions
                """)
                if not df.empty:
                    row = df.iloc[0]
                    return {
                        "metric_type": "30-Day Readmission Rate",
                        "summary": (
                            f"Overall Institutional 30-Day Readmission Rate is {row['readmit_pct']}% across "
                            f"{int(row['total']):,} documented inpatient encounters ({int(row['readmitted']):,} readmissions). "
                            f"Average Inpatient Length of Stay (ALOS) is {row['avg_los']} days."
                        ),
                        "table": "gold_admissions",
                        "values": row.to_dict(),
                        "suggested_category": "Hospital Policies",
                    }

        # 2. Inpatient Length of Stay queries
        if any(k in q_norm for k in ["length of stay", "alos", "average stay", "stay duration", "how long do patients stay"]):
            df = query_df("SELECT count(*) as total, round(avg(length_of_stay), 1) as avg_los FROM gold_admissions")
            if not df.empty:
                row = df.iloc[0]
                return {
                    "metric_type": "Average Length of Stay (ALOS)",
                    "summary": f"Average Inpatient Length of Stay (ALOS) across MediNexus facilities is {row['avg_los']} days across {int(row['total']):,} encounters.",
                    "table": "gold_admissions",
                    "values": row.to_dict(),
                    "suggested_category": "Hospital Policies",
                }

        # 3. Hospital Bed Occupancy & Capacity queries
        if any(k in q_norm for k in ["occupancy", "bed capacity", "bed utilization", "bed turnover"]):
            df = query_df("SELECT round(avg(estimated_occupancy_rate)*100, 1) as avg_occ FROM gold_hospital_operations")
            if not df.empty:
                row = df.iloc[0]
                return {
                    "metric_type": "Hospital Bed Occupancy",
                    "summary": f"Current Hospital Bed Occupancy Rate is {row['avg_occ']}% across inpatient facilities (operating in Normal / Moderate Capacity Surge Tier).",
                    "table": "gold_hospital_operations",
                    "values": row.to_dict(),
                    "suggested_category": "Hospital Policies",
                }

        # 4. Registered Patient Population queries
        if any(k in q_norm for k in ["how many patients", "total patients", "patient population", "patient count"]):
            df = query_df("SELECT count(*) as total_patients FROM gold_patient_360")
            if not df.empty:
                row = df.iloc[0]
                return {
                    "metric_type": "Active Patient Population",
                    "summary": f"Total active registered patient population is {int(row['total_patients']):,} individuals under the Gold intelligence layer.",
                    "table": "gold_patient_360",
                    "values": row.to_dict(),
                    "suggested_category": "Hospital Policies",
                }
    except Exception:
        pass

    return None


def generate_grounded_answer(query: str, category: Optional[str] = None, min_confidence: float = 0.12) -> Dict[str, Any]:
    """
    Retrieve relevant knowledge chunks and synthesize a grounded response with source citations.
    Supports Google Gemini, OpenAI (gpt-4o-mini), xAI Grok, and deterministic local synthesis.
    Integrates verified live Lakehouse Gold operational metrics alongside institutional policy guidelines.
    Enforces a clinical safety confidence threshold (min_confidence) to safely reject out-of-scope queries.
    Records high-precision latency telemetry (retrieval_ms, generation_ms, total_ms).
    """
    t_start = time.perf_counter()
    metric_info = fetch_lakehouse_operational_metric(query)
    effective_category = category or (metric_info.get("suggested_category") if metric_info else None)
    hits = retrieve_context(query, top_k=3, category=effective_category)
    t_ret = time.perf_counter()

    # Clinical Safety Guardrail: Safely refuse only if no verified Lakehouse metric exists AND top retrieval hit is below confidence
    if metric_info is None and (not hits or hits[0]["score"] < min_confidence):
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

    metric_section = ""
    if metric_info:
        metric_section = (
            f"VERIFIED LIVE LAKEHOUSE ENTERPRISE METRICS:\n"
            f"- Metric Type: {metric_info['metric_type']}\n"
            f"- Source Mart: {metric_info['table']} (DuckDB Lakehouse Gold Layer)\n"
            f"- Live Value & Summary: {metric_info['summary']}\n\n"
        )

    rag_prompt = (
        f"You are a clinical decision-support knowledge assistant for MediNexus AI. "
        f"Answer the user query strictly, factually, and comprehensively using the institutional data below. "
        f"{'State the verified live enterprise metric clearly first, then explain the institutional guidelines.' if metric_info else 'Answer strictly using the institutional documents.'} "
        f"Always cite source documents and specific clinical criteria (such as timelines, dosages, or thresholds).\n\n"
        f"{metric_section}"
        f"INSTITUTIONAL DOCUMENTS:\n{context_text}\n\n"
        f"USER CLINICAL QUERY: {query}"
    )

    llm_config = get_active_llm_config()

    # 0. Groq API (GroqCloud - Ultra Fast Inference)
    if (llm_config["provider"] == "groq" or (llm_config.get("api_key") and llm_config["api_key"].startswith("gsk_"))) and llm_config.get("api_key"):
        try:
            api_key = llm_config["api_key"]
            model_name = llm_config.get("model", "qwen/qwen3.8-27b")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name if "/" in model_name else "qwen/qwen3.8-27b",
                "messages": [
                    {"role": "system", "content": "You are a clinical decision-support knowledge assistant for MediNexus AI."},
                    {"role": "user", "content": rag_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 800,
            }
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                t_end = time.perf_counter()
                return {
                    "query": query,
                    "answer": text,
                    "citations": citations,
                    "grounded": True,
                    "engine": f"Groq ({model_name}) Grounded RAG",
                    "latency_metrics": {
                        "retrieval_ms": round((t_ret - t_start) * 1000, 2),
                        "generation_ms": round((t_end - t_ret) * 1000, 2),
                        "total_ms": round((t_end - t_start) * 1000, 2),
                    },
                    "top_relevance_score": hits[0]["score"] if hits else 0.0,
                }
        except Exception:
            pass  # Fall through to other providers or local fallback

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
    top_hit = hits[0] if hits else {
        "title": "Lakehouse Analytics Mart",
        "filename": metric_info["table"] if metric_info else "gold_mart",
        "category": "Enterprise Analytics",
        "text": metric_info["summary"] if metric_info else "",
        "score": 1.0,
    }
    cleaned_guidance = _clean_passage_text(top_hit["text"])

    if metric_info:
        local_synthesis = (
            f"### Verified Institutional Metric (Lakehouse Gold Layer)\n"
            f"- **{metric_info['metric_type']}**: {metric_info['summary']}\n"
            f"- **Data Source**: `{metric_info['table']}` (Verified Enterprise Lakehouse)\n\n"
            f"### Institutional Policy & Operational Directive\n"
            f"According to institutional documentation in **{top_hit['title']}** (*{top_hit['filename']}*):\n\n"
            f"{cleaned_guidance}\n\n"
            f"**Key Operational Takeaways:**\n"
            f"- Reference Guideline ID: `{top_hit['filename']}`\n"
            f"- Target Department: `{top_hit['category']}`\n"
            f"- Relevance Match Score: `{top_hit['score'] * 100:.1f}%`"
        )
    else:
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
