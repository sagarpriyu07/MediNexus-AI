"""
Base AI Agent Architecture for MediNexus AI.
Enforces tool-grounded reasoning, structured output separation (Facts, Predictions, Recommendations),
and multi-provider LLM integration (GroqCloud, Google Gemini, OpenAI, xAI Grok, or local deterministic synthesis).
"""

from typing import Dict, Any, List, Optional, Callable
import os
import requests
import json

from config.settings import get_active_llm_config
from src.security.audit import log_audit_event


class BaseHealthcareAgent:
    """
    Base class for MediNexus role-specific AI Agents.
    """

    def __init__(self, agent_name: str, role: str, system_prompt: str):
        self.agent_name = agent_name
        self.role = role
        self.system_prompt = system_prompt
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        """Register an analytical, ML, or database tool."""
        self.tools[name] = func

    def call_llm(self, prompt: str) -> Optional[str]:
        """
        Invoke active LLM provider (GroqCloud, Google Gemini, OpenAI, or xAI Grok)
        using the dynamically configured API key.
        """
        cfg = get_active_llm_config()
        if not cfg or not cfg.get("api_key"):
            return None

        api_key = cfg.get("api_key", "").strip()
        provider = cfg.get("provider", "")
        model = cfg.get("model", "")

        # 1. Groq (GroqCloud - Ultra Fast Inference)
        if provider == "groq" or api_key.startswith("gsk_"):
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                model_name = model if model and "/" in model else "qwen/qwen3.8-27b"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=12)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                return None

        # 2. Google Gemini API
        if provider == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-1.5-flash'}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": f"{self.system_prompt}\n\n{prompt}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
                }
                resp = requests.post(url, json=payload, timeout=12)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return None

        # 3. OpenAI API
        if provider == "openai":
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=12)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                return None

        # 4. xAI Grok API
        if provider == "grok":
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or "grok-beta",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                }
                resp = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=12)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                return None

        return None

    def synthesize_clinical_narrative(
        self,
        user_query: str,
        facts: List[str],
        predictions: List[str],
        recommendations: List[str],
    ) -> str:
        """
        Synthesize a natural, conversational executive/clinical copilot summary using the active LLM,
        or high-precision deterministic synthesis if offline.
        """
        facts_text = "\n".join([f"- {f}" for f in facts[:5]])
        preds_text = "\n".join([f"- {p}" for p in predictions[:3]])
        recs_text = "\n".join([f"- {r}" for r in recommendations[:4]])

        prompt = (
            f"USER INQUIRY: {user_query}\n\n"
            f"VERIFIED CLINICAL/OPERATIONAL LAKEHOUSE FACTS:\n{facts_text}\n\n"
            f"PREDICTIVE ML RISK FORECASTS:\n{preds_text}\n\n"
            f"PRESCRIPTIVE ACTION PROTOCOLS:\n{recs_text}\n\n"
            f"Provide a concise, expert clinical/operational copilot summary (1-2 clear paragraphs) "
            f"directly answering the user. Highlight critical risk findings and priority actions."
        )

        llm_resp = self.call_llm(prompt)
        if llm_resp:
            return llm_resp.strip()

        # Deterministic offline narrative fallback
        fact_hl = facts[0] if facts else "Baseline telemetry retrieved from Gold layer."
        pred_hl = predictions[0] if predictions else "Risk within standard monitoring thresholds."
        rec_hl = recommendations[0] if recommendations else "Follow standard institutional directives."

        return (
            f"**{self.agent_name} Analysis:** {fact_hl} {pred_hl} "
            f"Priority Action: {rec_hl}"
        )

    def format_structured_response(
        self,
        facts: List[str],
        predictions: List[str],
        recommendations: List[str],
        evidence: List[str],
        raw_context: Dict[str, Any] = None,
        notes: str = "",
        text_answer: str = "",
    ) -> Dict[str, Any]:
        """
        Structure agent responses into distinct, auditable sections:
        FACTS, PREDICTIONS, RECOMMENDATIONS, EVIDENCE, plus conversational TEXT_ANSWER.
        """
        return {
            "agent": self.agent_name,
            "role": self.role,
            "text_answer": text_answer,
            "facts": facts,
            "predictions": predictions,
            "recommendations": recommendations,
            "evidence": evidence,
            "notes": notes,
            "raw_context": raw_context or {},
        }
