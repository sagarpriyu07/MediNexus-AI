"""
Authentication and Role Selector Login Page for MediNexus AI.
Provides:
1. Instant 1-Click Role Login for smooth, frictionless executive demos.
2. Traditional Username & Password form for manual credential verification.
Data Engineer is the #1 persona. No individual personal names are displayed.
"""

import streamlit as st
from config.settings import DEMO_USERS, DEMO_PASSWORD_DEFAULT
from config.roles import ROLE_PERMISSIONS
from src.security.auth import authenticate_user
from src.security.audit import log_audit_event


def render_login_page():
    """Render minimalist, high-contrast login interface with 1-Click and Manual options."""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 22px;">
            <div style="display: inline-block; background: #0F172A; color: #2DD4BF; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">
                ROLE-BASED ACCESS CONTROL PORTAL
            </div>
            <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 4px; font-size: 1.85rem;">MediNexus Access Portal</h2>
            <p style="color: #64748B; font-size: 0.92rem; max-width: 600px; margin: 0 auto;">
                Select a role for instant demonstration or sign in with credentials.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab_quick, tab_manual = st.tabs([
        "⚡ 1-Click Role Login",
        "🔐 Manual Credentials",
    ])

    # =========================================================================
    # TAB 1: 1-CLICK ROLE LOGIN (DATA ENGINEER FIRST, NO PERSONAL NAMES)
    # =========================================================================
    with tab_quick:
        personas = [
            {
                "key": "data_engineer",
                "role": "Data Engineer",
                "icon": "⚙️",
                "title": "Data Engineer",
                "dept": "Data Platform & Engineering",
                "scope": "Medallion Lakehouse (Bronze ➔ Silver ➔ Gold), 99.6% Data Quality index, and ML model training.",
                "color": "#0D9488",
                "btn_label": "Enter as Data Engineer",
            },
            {
                "key": "dr_chen",
                "role": "Doctor",
                "icon": "🩺",
                "title": "Doctor",
                "dept": "Internal Medicine & Cardiology",
                "scope": "Patient 360, 30-day readmission risk, length-of-stay predictions, and MediCare Copilot.",
                "color": "#0284C7",
                "btn_label": "Enter as Doctor",
            },
            {
                "key": "admin_holloway",
                "role": "Hospital Administrator",
                "icon": "🏢",
                "title": "Hospital Administrator",
                "dept": "Executive Operations",
                "scope": "Bed occupancy metrics, Level 1-4 Bed Surge alerts, readmission root causes, and HealthAnalyst Copilot.",
                "color": "#0F172A",
                "btn_label": "Enter as Hospital Administrator",
            },
            {
                "key": "pharmacist_patel",
                "role": "Pharmacist",
                "icon": "💊",
                "title": "Pharmacist",
                "dept": "Central Clinical Pharmacy",
                "scope": "Formulary stockout alerts, 30-day medication demand forecaster, auto-POs, and PharmaLab Copilot.",
                "color": "#D97706",
                "btn_label": "Enter as Pharmacist",
            },
            {
                "key": "lab_tech_kim",
                "role": "Laboratory Specialist",
                "icon": "🔬",
                "title": "Laboratory Specialist",
                "dept": "Pathology & Diagnostics",
                "scope": "Analyzer throughput, turnaround times (TAT), and critical panic value callback alerts.",
                "color": "#7C3AED",
                "btn_label": "Enter as Laboratory Specialist",
            },
            {
                "key": "receptionist_davis",
                "role": "Receptionist",
                "icon": "📋",
                "title": "Receptionist",
                "dept": "Outpatient Scheduling",
                "scope": "Intake queues, physician appointment scheduling, and patient check-in (HIPAA compliant).",
                "color": "#059669",
                "btn_label": "Enter as Receptionist",
            },
            {
                "key": "it_admin_torvalds",
                "role": "IT Administrator",
                "icon": "🛡️",
                "title": "IT Administrator",
                "dept": "Security & Infrastructure",
                "scope": "Immutable dual audit logs, RBAC enforcement, session monitoring, and DuckDB storage telemetry.",
                "color": "#DC2626",
                "btn_label": "Enter as IT Administrator",
            },
        ]

        def login_as_persona(username_key):
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
                    action="LOGIN_1CLICK",
                    resource="SYSTEM_AUTH",
                    status="SUCCESS",
                    details=f"Demo 1-Click Login as {user_profile['role']}",
                )

                role_meta = ROLE_PERMISSIONS.get(user_profile["role"], {})
                st.session_state["current_page"] = role_meta.get("default_page", "welcome")
                if username_key == "data_engineer" or user_profile.get("role") == "Data Engineer":
                    st.session_state["pipeline_completed"] = False
                st.rerun()

        # Render personas in 2 clean columns
        c_left, c_right = st.columns(2)
        for idx, p in enumerate(personas):
            target_col = c_left if idx % 2 == 0 else c_right
            with target_col:
                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {p['color']};
                                border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div style="font-weight: 700; color: #0F172A; font-size: 0.98rem;">
                                <span style="margin-right: 6px;">{p['icon']}</span>{p['title']}
                            </div>
                            <span style="background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">
                                {p['dept']}
                            </span>
                        </div>
                        <div style="color: #475569; font-size: 0.82rem; line-height: 1.4; margin-bottom: 10px;">
                            {p['scope']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(p["btn_label"], key=f"btn_1click_{p['key']}", use_container_width=True, type="primary" if idx == 0 else "secondary"):
                    login_as_persona(p["key"])

    # =========================================================================
    # TAB 2: MANUAL USERNAME & PASSWORD LOGIN
    # =========================================================================
    with tab_manual:
        col_m_left, col_m_center, col_m_right = st.columns([1, 2, 1])
        with col_m_center:
            persona_options = {
                "Prefill role credentials...": None,
                "⚙️ Data Engineer": "data_engineer",
                "🩺 Doctor": "dr_chen",
                "🏢 Hospital Administrator": "admin_holloway",
                "💊 Pharmacist": "pharmacist_patel",
                "🔬 Laboratory Specialist": "lab_tech_kim",
                "📋 Receptionist": "receptionist_davis",
                "🛡️ IT Administrator": "it_admin_torvalds",
            }
            selected_prefill = st.selectbox("Optional: Prefill Credentials", options=list(persona_options.keys()), index=0)
            prefill_user = persona_options.get(selected_prefill)

            default_u = prefill_user if prefill_user else ""
            default_p = DEMO_PASSWORD_DEFAULT if prefill_user else ""

            with st.form("manual_login_form"):
                username_input = st.text_input("Username", value=default_u)
                password_input = st.text_input("Password", value=default_p, type="password")
                submit = st.form_submit_button("Sign In with Credentials", type="primary", use_container_width=True)

                if submit:
                    if not username_input or not password_input:
                        st.error("Please enter both username and password.")
                    else:
                        user_profile = authenticate_user(username_input, password_input)
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
                                action="LOGIN_MANUAL",
                                resource="SYSTEM_AUTH",
                                status="SUCCESS",
                                details="Authenticated successfully via manual form.",
                            )

                            role_meta = ROLE_PERMISSIONS.get(user_profile["role"], {})
                            st.session_state["current_page"] = role_meta.get("default_page", "welcome")
                            if user_profile.get("role") == "Data Engineer":
                                st.session_state["pipeline_completed"] = False
                            st.success(f"Authenticated as {user_profile['role']}!")
                            st.rerun()
                        else:
                            log_audit_event(
                                username=username_input,
                                role="Unknown",
                                action="LOGIN_FAILED",
                                resource="SYSTEM_AUTH",
                                status="FAILED",
                                details="Invalid credentials provided.",
                            )
                            st.error("Invalid credentials. Please verify your username and password.")

            # Credentials reference table
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 View Demo Accounts & Roles"):
                st.markdown(f"""
                | Role | Username | Password |
                | :--- | :--- | :--- |
                | **Data Engineer** | `data_engineer` | `{DEMO_PASSWORD_DEFAULT}` |
                | **Doctor** | `dr_chen` | `{DEMO_PASSWORD_DEFAULT}` |
                | **Hospital Administrator** | `admin_holloway` | `{DEMO_PASSWORD_DEFAULT}` |
                | **Pharmacist** | `pharmacist_patel` | `{DEMO_PASSWORD_DEFAULT}` |
                | **Laboratory Specialist** | `lab_tech_kim` | `{DEMO_PASSWORD_DEFAULT}` |
                | **Receptionist** | `receptionist_davis` | `{DEMO_PASSWORD_DEFAULT}` |
                | **IT Administrator** | `it_admin_torvalds` | `{DEMO_PASSWORD_DEFAULT}` |
                """)


if __name__ == "__main__":
    render_login_page()
