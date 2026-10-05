"""
Welcome Page for MediNexus AI.
Presents the platform vision, data-to-decision architecture, and role-based entrypoints.
Data Engineer is the #1 entrypoint. Personal names are removed.
"""

import streamlit as st
from config.settings import DEMO_USERS
from config.roles import ROLE_PERMISSIONS
from src.security.audit import log_audit_event


def render_welcome_page():
    """Render minimalist high-impact landing page."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0D9488 100%);
                    padding: 28px 32px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);">
            <span style="background: rgba(13, 148, 136, 0.3); border: 1px solid #14B8A6; color: #5EEAD4;
                         padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">
                HEALTHCARE DATA-TO-DECISION PLATFORM
            </span>
            <h1 style="color: white; margin-top: 8px; margin-bottom: 6px; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                MediNexus AI
            </h1>
            <p style="color: #CBD5E1; font-size: 1rem; max-width: 680px; line-height: 1.45; margin-bottom: 0;">
                Role-based clinical intelligence transforming heterogeneous healthcare data into trusted, predictive, and prescriptive decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Architectural Philosophy Banner
    st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center; flex-wrap: wrap; gap: 6px;">
                <div><b style="color: #0D9488; font-size: 0.95rem;">DATA</b><br><span style="color: #64748B; font-size: 0.72rem;">Heterogeneous</span></div>
                <div style="color: #94A3B8;">➔</div>
                <div><b style="color: #0284C7; font-size: 0.95rem;">TRUST</b><br><span style="color: #64748B; font-size: 0.72rem;">Medallion 99.6%</span></div>
                <div style="color: #94A3B8;">➔</div>
                <div><b style="color: #6366F1; font-size: 0.95rem;">INTELLIGENCE</b><br><span style="color: #64748B; font-size: 0.72rem;">Gold Marts</span></div>
                <div style="color: #94A3B8;">➔</div>
                <div><b style="color: #D97706; font-size: 0.95rem;">PREDICTION</b><br><span style="color: #64748B; font-size: 0.72rem;">ML Forecasts</span></div>
                <div style="color: #94A3B8;">➔</div>
                <div><b style="color: #10B981; font-size: 0.95rem;">PRESCRIPTION</b><br><span style="color: #64748B; font-size: 0.72rem;">Directives</span></div>
                <div style="color: #94A3B8;">➔</div>
                <div><b style="color: #EF4444; font-size: 0.95rem;">ACTION</b><br><span style="color: #64748B; font-size: 0.72rem;">Role Consoles</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1-Click Role Selector (Data Engineer FIRST, NO personal names)
    st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Select a Role to Enter Console:</p>", unsafe_allow_html=True)

    def launch_role(username_key):
        user_profile = DEMO_USERS.get(username_key)
        if user_profile:
            st.session_state["authenticated"] = True
            st.session_state["user"] = user_profile
            st.session_state["username"] = user_profile["username"]
            st.session_state["name"] = user_profile["name"]
            st.session_state["role"] = user_profile["role"]
            st.session_state["department"] = user_profile["department"]

            log_audit_event(
                username=user_profile["username"],
                role=user_profile["role"],
                action="LAUNCH_WELCOME",
                resource="SYSTEM_AUTH",
                status="SUCCESS",
                details=f"1-Click Launch from Welcome as {user_profile['role']}",
            )

            role_meta = ROLE_PERMISSIONS.get(user_profile["role"], {})
            st.session_state["current_page"] = role_meta.get("default_page", "welcome")
            if username_key == "data_engineer" or user_profile.get("role") == "Data Engineer":
                st.session_state["pipeline_completed"] = False
            st.rerun()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        if st.button("⚙️ Data Engineer", key="w_btn_de", use_container_width=True, type="primary"):
            launch_role("data_engineer")
    with r2:
        if st.button("🩺 Doctor", key="w_btn_doc", use_container_width=True):
            launch_role("dr_chen")
    with r3:
        if st.button("🏢 Administrator", key="w_btn_adm", use_container_width=True):
            launch_role("admin_holloway")
    with r4:
        if st.button("💊 Pharmacist", key="w_btn_phm", use_container_width=True):
            launch_role("pharmacist_patel")

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        if st.button("🔬 Laboratory", key="w_btn_lab", use_container_width=True):
            launch_role("lab_tech_kim")
    with r6:
        if st.button("📋 Receptionist", key="w_btn_rec", use_container_width=True):
            launch_role("receptionist_davis")
    with r7:
        if st.button("🛡️ IT Security", key="w_btn_it", use_container_width=True):
            launch_role("it_admin_torvalds")
    with r8:
        if st.button("🔑 Access Portal", key="w_btn_login", use_container_width=True):
            st.session_state["current_page"] = "login"
            st.rerun()

    st.markdown("---")

    # Concise Architecture Breakdown
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 0.95rem;">Data Engineering / Producer Layer</h4>
                <p style="color: #64748B; font-size: 0.84rem; line-height: 1.45; margin: 0;">
                    Orchestrates Bronze Parquet ingestion, Silver deduplication and 99.6% quality scoring, Gold Star Marts in DuckDB, and predictive ML model training.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 0.95rem;">Role-Based Consumption Layer</h4>
                <p style="color: #64748B; font-size: 0.84rem; line-height: 1.45; margin: 0;">
                    Consumes strictly validated Gold intelligence across 7 dedicated consoles with specialized AI Copilots, prescriptive rules, and zero clinical leakage.
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_welcome_page()
