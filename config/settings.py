"""
System Settings and Environment Configuration for MediNexus AI.
"""

import os
from pathlib import Path
from typing import Dict, Any

from config.constants import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    BRONZE_DATA_DIR,
    SILVER_DATA_DIR,
    GOLD_DATA_DIR,
    METADATA_DIR,
    ML_RESULTS_DIR,
    DATABASE_DIR,
    DATABASE_PATH,
    MODELS_DIR,
    DOCUMENTS_DIR,
    LOGS_DIR,
)

# Application Metadata
APP_NAME = "MediNexus AI"
APP_TAGLINE = "Role-Based Healthcare Data-to-Decision Intelligence Platform"
APP_VERSION = "2.4.0"
RANDOM_SEED = 42

# Environment Variables Loading (Simple .env parser without external dependencies)
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Check Streamlit Cloud secrets if running hosted on Streamlit Cloud
_st_secrets = {}
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        _st_secrets = dict(st.secrets)
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or str(_st_secrets.get("GEMINI_API_KEY", ""))
ENABLE_GEMINI = bool(GEMINI_API_KEY and len(GEMINI_API_KEY) > 10)

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or str(_st_secrets.get("OPENAI_API_KEY", ""))
ENABLE_OPENAI = bool(OPENAI_API_KEY and len(OPENAI_API_KEY) > 10)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or str(_st_secrets.get("OPENAI_MODEL", "gpt-4o-mini"))

# xAI Grok API Configuration
GROK_API_KEY = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "") or str(_st_secrets.get("GROK_API_KEY", ""))
ENABLE_GROK = bool(GROK_API_KEY and len(GROK_API_KEY) > 10)
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta") or str(_st_secrets.get("GROK_MODEL", "grok-beta"))

# Dynamic Active Provider Detection and Session Management
def get_active_llm_config() -> Dict[str, Any]:
    """
    Dynamically retrieve active LLM provider configuration from Streamlit session state,
    environment variables, or persistent .env settings.
    """
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            override_provider = st.session_state.get("llm_provider_override")
            custom_key = st.session_state.get("llm_api_key_override", "").strip()
            if override_provider == "gemini" and custom_key:
                return {
                    "provider": "gemini",
                    "provider_name": "Gemini 1.5 Flash",
                    "api_key": custom_key,
                    "model": "gemini-1.5-flash",
                    "is_cloud": True,
                }
            elif override_provider == "openai" and custom_key:
                model_name = st.session_state.get("llm_model_override", "gpt-4o-mini")
                return {
                    "provider": "openai",
                    "provider_name": f"OpenAI ({model_name})",
                    "api_key": custom_key,
                    "model": model_name,
                    "is_cloud": True,
                }
            elif override_provider == "grok" and custom_key:
                return {
                    "provider": "grok",
                    "provider_name": f"Grok ({GROK_MODEL})",
                    "api_key": custom_key,
                    "model": GROK_MODEL,
                    "is_cloud": True,
                }
            elif override_provider == "local":
                return {
                    "provider": "local",
                    "provider_name": "Deterministic Local Engine",
                    "api_key": "",
                    "model": "local",
                    "is_cloud": False,
                }
    except Exception:
        pass

    grok_k = os.getenv("GROK_API_KEY", "") or GROK_API_KEY
    if grok_k and len(grok_k) > 10:
        return {
            "provider": "grok",
            "provider_name": f"Grok ({GROK_MODEL})",
            "api_key": grok_k,
            "model": GROK_MODEL,
            "is_cloud": True,
        }

    gemini_k = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if gemini_k and len(gemini_k) > 10:
        return {
            "provider": "gemini",
            "provider_name": "Gemini 1.5 Flash",
            "api_key": gemini_k,
            "model": "gemini-1.5-flash",
            "is_cloud": True,
        }

    openai_k = os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY
    if openai_k and len(openai_k) > 10:
        return {
            "provider": "openai",
            "provider_name": f"OpenAI ({OPENAI_MODEL})",
            "api_key": openai_k,
            "model": OPENAI_MODEL,
            "is_cloud": True,
        }

    return {
        "provider": "local",
        "provider_name": "Deterministic Local Engine",
        "api_key": "",
        "model": "local",
        "is_cloud": False,
    }


def save_llm_api_key(provider: str, api_key: str, model: str = "") -> None:
    """Save an API key to session state, os.environ, and write to .env file for persistence."""
    api_key = api_key.strip()
    try:
        import streamlit as st
        st.session_state["llm_provider_override"] = provider
        st.session_state["llm_api_key_override"] = api_key
        if model:
            st.session_state["llm_model_override"] = model
    except Exception:
        pass

    env_key = "GEMINI_API_KEY" if provider == "gemini" else ("OPENAI_API_KEY" if provider == "openai" else "GROK_API_KEY")
    os.environ[env_key] = api_key
    if model:
        if provider == "openai":
            os.environ["OPENAI_MODEL"] = model
        elif provider == "grok":
            os.environ["GROK_MODEL"] = model

    # Persist to .env file safely
    try:
        lines = []
        if ENV_PATH.exists():
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{env_key}="):
                new_lines.append(f"{env_key}={api_key}\n")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{env_key}={api_key}\n")

        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        pass


def clear_llm_api_key() -> None:
    """Revert to Deterministic Local Engine and clear overrides."""
    try:
        import streamlit as st
        st.session_state["llm_provider_override"] = "local"
        st.session_state["llm_api_key_override"] = ""
    except Exception:
        pass
    for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "GROK_API_KEY"]:
        if k in os.environ:
            del os.environ[k]


ACTIVE_LLM_PROVIDER = get_active_llm_config()["provider_name"]

# Preconfigured Demo Users
# Note: In production, passwords are stored as secure hashes.
# Default password for all demo accounts is: "MediNexus@2026"
DEMO_PASSWORD_DEFAULT = "MediNexus@2026"

DEMO_USERS: Dict[str, Dict[str, Any]] = {
    "data_engineer": {
        "username": "data_engineer",
        "name": "Data Engineer",
        "email": "data.engineer@medinexus.org",
        "role": "Data Engineer",
        "department": "Data Platform & Engineering",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "dr_chen": {
        "username": "dr_chen",
        "name": "Doctor",
        "email": "doctor@medinexus.org",
        "role": "Doctor",
        "department": "Internal Medicine & Cardiology",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "admin_holloway": {
        "username": "admin_holloway",
        "name": "Hospital Administrator",
        "email": "admin@medinexus.org",
        "role": "Hospital Administrator",
        "department": "Executive Operations",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "pharmacist_patel": {
        "username": "pharmacist_patel",
        "name": "Pharmacist",
        "email": "pharmacist@medinexus.org",
        "role": "Pharmacist",
        "department": "Central Clinical Pharmacy",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "lab_tech_kim": {
        "username": "lab_tech_kim",
        "name": "Laboratory Specialist",
        "email": "lab@medinexus.org",
        "role": "Laboratory Technician",
        "department": "Pathology & Diagnostics",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "receptionist_davis": {
        "username": "receptionist_davis",
        "name": "Receptionist",
        "email": "receptionist@medinexus.org",
        "role": "Receptionist",
        "department": "Outpatient Registration & Care Flow",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
    "it_admin_torvalds": {
        "username": "it_admin_torvalds",
        "name": "IT Administrator",
        "email": "it.admin@medinexus.org",
        "role": "IT Administrator",
        "department": "Information Security & Infrastructure",
        "plain_pwd": DEMO_PASSWORD_DEFAULT,
    },
}


def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        BRONZE_DATA_DIR,
        SILVER_DATA_DIR,
        GOLD_DATA_DIR,
        METADATA_DIR,
        ML_RESULTS_DIR,
        DATABASE_DIR,
        MODELS_DIR,
        DOCUMENTS_DIR,
        LOGS_DIR,
        DOCUMENTS_DIR / "hospital_policies",
        DOCUMENTS_DIR / "clinical_guidelines",
        DOCUMENTS_DIR / "pharmacy",
        DOCUMENTS_DIR / "laboratory",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


ensure_directories()
