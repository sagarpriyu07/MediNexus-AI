"""
Base AI Agent Architecture for MediNexus AI.
Enforces tool-grounded reasoning, structured output separation (Facts, Predictions, Recommendations),
and integration with Gemini API or deterministic offline analytical synthesis.
"""

from typing import Dict, Any, List, Optional, Callable
import os
import requests
import json

from config.settings import GEMINI_API_KEY, ENABLE_GEMINI
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

    def call_gemini(self, prompt: str) -> Optional[str]:
        """Invoke Gemini REST API if configured."""
        if not ENABLE_GEMINI:
            return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{self.system_prompt}\n\n{prompt}"}
                        ]
                    }
                ]
            }
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None
        return None

    def format_structured_response(
        self,
        facts: List[str],
        predictions: List[str],
        recommendations: List[str],
        evidence: List[str],
        raw_context: Dict[str, Any] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Structure agent responses into distinct, auditable sections:
        FACTS, PREDICTIONS, RECOMMENDATIONS, EVIDENCE.
        """
        return {
            "agent": self.agent_name,
            "role": self.role,
            "facts": facts,
            "predictions": predictions,
            "recommendations": recommendations,
            "evidence": evidence,
            "notes": notes,
            "raw_context": raw_context or {},
        }
