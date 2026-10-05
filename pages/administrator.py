"""
Hospital Executive & Administrator Dashboard for MediNexus AI.
Consumes Gold layer models (gold_hospital_operations, gold_admissions, gold_billing, gold_patient_360),
diagnostic readmission drivers, occupancy forecasts, surge capacity plans, and the HealthAnalyst AI Agent.
"""

import streamlit as st
import pandas as pd

from src.analytics.metrics import compute_executive_kpis
from src.analytics.descriptive import get_admissions_trend_chart, get_department_revenue_chart
from src.analytics.diagnostic import diagnose_readmissions
from src.prescriptive.hospital_rules import get_hospital_capacity_plan
from src.agents.healthanalyst_agent import HealthAnalystAgent
from src.utils.chat_ui import render_ai_chatbot
from src.utils.database import query_df
from src.utils.helpers import format_number, format_currency, PALETTE
from src.security.audit import log_audit_event


def render_administrator_page():
    """Render Hospital Administrator executive dashboard."""
    user = st.session_state.get("user", {})
    username = user.get("username", "admin_holloway")

    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h2 style="color: #0F172A; font-weight: 800; margin: 0;">Hospital Executive Operations Console</h2>
                <p style="color: #64748B; margin: 0; font-size: 0.92rem;">
                    Enterprise KPIs, Readmission Diagnostics, Bed Capacity Forecasting, and HealthAnalyst Copilot.
                </p>
            </div>
            <span style="background: #0F172A; color: white; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 0.82rem;">
                GOLD ENTERPRISE MART
            </span>
        </div>
    """, unsafe_allow_html=True)

    df_check = query_df("SELECT COUNT(*) as cnt FROM gold_hospital_operations")
    if df_check.empty or int(df_check.iloc[0]["cnt"]) == 0:
        st.info("Lakehouse gold models are awaiting ingestion. Please run the Medallion Pipeline in Data Engineering.")
        return

    # 1. Executive Performance KPIs
    kpis = compute_executive_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Inpatient Admissions", format_number(kpis["total_admissions"]))
    c2.metric("System Bed Occupancy", f"{kpis['occupancy_rate_pct']}%", delta="Target: 80-85%")
    c3.metric("30-Day Readmission Rate", f"{kpis['readmission_rate_pct']}%", delta="Benchmark <= 15%", delta_color="inverse")
    c4.metric("Average Length of Stay", f"{kpis['average_los_days']} Days")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Billed Revenue", format_currency(kpis["total_revenue"]))
    c6.metric("Collection Efficiency", f"{kpis['collection_rate_pct']}%")
    c7.metric("Active Patient Population", format_number(kpis["total_patients"]))
    c8.metric("Operating Hospital Units", "5 Facilities")

    st.markdown("---")

    # 2. Longitudinal Charts
    col_adm_chart, col_rev_chart = st.columns(2)
    with col_adm_chart:
        st.plotly_chart(get_admissions_trend_chart(), use_container_width=True)

    with col_rev_chart:
        st.plotly_chart(get_department_revenue_chart(), use_container_width=True)

    st.markdown("---")

    # 3. Diagnostic Readmission Analytics & Prescriptive Surge Plan
    col_diag, col_surge = st.columns([1, 1])

    with col_diag:
        st.markdown("### 🔍 Diagnostic Analytics: Why Readmissions Occur")
        diag_res = diagnose_readmissions()

        st.markdown(f"**Diagnostic Summary:** {diag_res['summary']}")
        st.markdown("##### Statistically Correlated Contributing Factors:")
        for factor in diag_res["associated_factors"]:
            st.markdown(f"- ⚠️ {factor}")

        with st.expander("View Readmissions by Admission Type"):
            st.dataframe(pd.DataFrame(diag_res["by_admission_type"]), use_container_width=True, hide_index=True)

        with st.expander("View Readmissions by Comorbidity Burden"):
            st.dataframe(pd.DataFrame(diag_res["by_comorbidity_burden"]), use_container_width=True, hide_index=True)

    with col_surge:
        st.markdown("### 🏥 Prescriptive Bed Surge & Capacity Directive")
        capacity_plan = get_hospital_capacity_plan(
            occupancy_rate=float(kpis["occupancy_rate_pct"]) / 100.0,
            projected_daily_admissions=int(kpis["total_admissions"] // 90),
            average_los=float(kpis["average_los_days"]),
        )

        st.markdown(f"""
            <div style="background: #FEF2F2; border: 1px solid #FECACA; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #991B1B;">CURRENT CAPACITY SURGE LEVEL</span>
                <h3 style="color: #B91C1C; margin: 4px 0;">{capacity_plan['surge_tier']}</h3>
                <span style="font-size: 0.85rem; color: #7F1D1D;">{capacity_plan['throughput_assessment']}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Executive Action Directives:")
        for directive in capacity_plan["action_directives"]:
            st.markdown(f"- ✅ {directive}")

    st.markdown("---")

    # 4. Interactive HealthAnalyst AI Copilot (Modern Chatbot UI)
    st.markdown("---")
    render_ai_chatbot(
        agent_instance=HealthAnalystAgent(),
        chat_key="admin_healthanalyst",
        title="HealthAnalyst AI Executive Copilot",
        subtitle="Gold Enterprise Marts & Operational Policies Grounded",
        suggested_prompts=[
            "Why did readmissions increase across inpatient wards?",
            "Provide bed capacity and surge forecast",
            "What are institutional inpatient discharge guidelines?",
        ],
        username=username,
    )


if __name__ == "__main__":
    if not st.session_state.get("authenticated"):
        from config.settings import DEMO_USERS
        st.session_state["authenticated"] = True
        st.session_state["user"] = DEMO_USERS["admin_holloway"]
        st.session_state["username"] = "admin_holloway"
        st.session_state["name"] = "Hospital Administrator"
        st.session_state["role"] = "Hospital Administrator"
    render_administrator_page()

