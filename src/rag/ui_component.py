"""
Clinical RAG & Knowledge Navigator UI Component for MediNexus AI.
Provides a modern, high-contrast, interactive search and decision-support interface for
retrieving and synthesizing answers from institutional healthcare guidelines.
"""

import streamlit as st
from typing import Optional, List
from datetime import datetime

from src.rag.knowledge_base import generate_grounded_answer
from config.settings import get_active_llm_config, save_llm_api_key, clear_llm_api_key


SAMPLE_RAG_QUERIES = [
    "What are the discharge criteria for chronic heart failure patients?",
    "What is the follow-up protocol for high readmission risk patients?",
    "What are the HIPAA patient privacy safeguards under minimum necessary access?",
    "What are the storage temperature requirements for refrigerated medications?",
    "What personal protective equipment is required for laboratory biosafety?",
    "What is the appointment cancellation and late arrival policy?",
]


def render_rag_navigator(key_prefix: str = "main_rag"):
    """
    Render the interactive Clinical RAG Navigator console with query chips,
    grounded synthesis, cited excerpts, and active LLM engine observability.
    """
    llm_cfg = get_active_llm_config()

    st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h3 style="margin: 0; color: #0F172A; font-weight: 800; font-size: 1.25rem;">
                        🤖 Clinical RAG Knowledge Navigator
                    </h3>
                    <p style="margin: 4px 0 0 0; color: #64748B; font-size: 0.88rem;">
                        Interactive grounded question answering against verified institutional policies and guidelines.
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {'#EFF6FF' if llm_cfg['is_cloud'] else '#F0FDF4'};
                                 color: {'#1D4ED8' if llm_cfg['is_cloud'] else '#15803D'};
                                 border: 1px solid {'#BFDBFE' if llm_cfg['is_cloud'] else '#BBF7D0'};
                                 padding: 4px 12px; border-radius: 16px; font-size: 0.78rem; font-weight: 700;">
                        ● Active Engine: {llm_cfg['provider_name']}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Quick Suggestion Query Chips
    st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #334155; margin-bottom: 6px;'>Suggested Clinical & Operational Inquiries:</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    clicked_chip = None

    with c1:
        if st.button("❤️ Heart Failure Discharge", key=f"{key_prefix}_chip_1", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[0]
        if st.button("💊 Refrigerated Drug Storage", key=f"{key_prefix}_chip_4", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[3]

    with c2:
        if st.button("📋 High-Risk Readmission Protocol", key=f"{key_prefix}_chip_2", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[1]
        if st.button("🧪 Lab Biosafety & PPE", key=f"{key_prefix}_chip_5", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[4]

    with c3:
        if st.button("🔒 HIPAA Privacy Safeguards", key=f"{key_prefix}_chip_3", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[2]
        if st.button("⏰ Appointment Cancellation Rules", key=f"{key_prefix}_chip_6", use_container_width=True):
            clicked_chip = SAMPLE_RAG_QUERIES[5]

    st.markdown("<br style='line-height: 8px;'>", unsafe_allow_html=True)

    # Input Form
    active_query = clicked_chip
    with st.form(key=f"{key_prefix}_form", clear_on_submit=False):
        user_input = st.text_input(
            "Enter clinical query or policy inquiry:",
            value=clicked_chip if clicked_chip else "",
            placeholder="e.g. What are the rules for inpatient discharge departure times and multidisciplinary rounds?",
            key=f"{key_prefix}_query_input",
        )
        sub_col1, sub_col2 = st.columns([1, 4])
        with sub_col1:
            submit_btn = st.form_submit_button("🔍 Search & Synthesize", type="primary", use_container_width=True)

    query_to_run = user_input.strip() if submit_btn else clicked_chip

    if query_to_run:
        with st.spinner(f"Retrieving institutional guidelines and synthesizing answer using {llm_cfg['provider_name']}..."):
            res = generate_grounded_answer(query_to_run)

        is_refusal = res.get("safe_refusal", False) or res.get("out_of_scope", False)
        lat = res.get("latency_metrics", {})
        tot_ms = lat.get("total_ms", 1.5)
        ret_ms = lat.get("retrieval_ms", 1.0)
        engine_name = res.get("engine", llm_cfg["provider_name"])

        # Result Presentation Box
        if is_refusal:
            st.warning(f"🛡️ **Clinical Safety Guardrail Activated** (Low Confidence / Out-of-Scope Query)")
            st.markdown(f"""
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 16px; margin: 10px 0;">
                    <p style="color: #92400E; font-size: 0.94rem; line-height: 1.5; margin: 0;">
                        {res.get('answer')}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 18px; margin: 12px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: #166534; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                            ✅ Verified Grounded Answer
                        </span>
                        <span style="font-size: 0.78rem; color: #475569; font-weight: 600;">
                            ⚡ Total Latency: <b>{tot_ms} ms</b> (Vector Retrieval: {ret_ms} ms)
                        </span>
                    </div>
                    <div style="color: #0F172A; font-size: 0.95rem; line-height: 1.6;">
                        {res.get('answer')}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Citations and Source Evidence Accordion
        citations = res.get("citations", [])
        if citations:
            with st.expander(f"📚 Retrieved Source Evidence & Policy Documents ({len(citations)} Chunks Matched)", expanded=True):
                for idx, c in enumerate(citations):
                    relevance_pct = round(c.get("relevance", 0.0) * 100, 1)
                    st.markdown(f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b style="color: #0F172A; font-size: 0.9rem;">{idx + 1}. {c.get('title')}</b>
                                <span style="background: #E0F2FE; color: #0369A1; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">
                                    {relevance_pct}% Match
                                </span>
                            </div>
                            <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">
                                File: <code>{c.get('filename')}</code> | Department: <b>{c.get('category')}</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
