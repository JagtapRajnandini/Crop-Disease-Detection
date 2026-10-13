"""
pages/Farm_History.py
---------------------
Screen 2: Farm Scan Audit Trail & Outbreak Analytics
(Protected: Only accessible when logged in)
"""

import streamlit as st
import pandas as pd
from src.auth.database import get_user_scans

st.set_page_config(page_title="Farm History — AgriScan AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAF8; color: #1B382B; }

    .metric-card {
        background: #FFFFFF; border-radius: 14px; padding: 20px;
        border-left: 5px solid #2D6A4F; box-shadow: 0 4px 12px rgba(27,67,50,0.04); height: 100%;
    }
    .metric-card .label { color: #52796F; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; }
    .metric-card .value { color: #1B4332; font-size: 1.4rem; font-weight: 800; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Auth Guard: Restrict access if not logged in ───────────────────────────────
if "user" not in st.session_state or not st.session_state.user:
    st.markdown("<h2 style='color: #1B4332; font-weight: 800;'>🔒 Access Restricted</h2>", unsafe_allow_html=True)
    st.warning("You must be logged in to view your private farm audit logs and outbreak analytics.")
    if st.button("← Go to Home Page to Sign In", type="primary"):
        st.switch_page("app.py")
    st.stop()

# Top Bar
u = st.session_state.user
c_h1, c_h2 = st.columns([6, 2])
with c_h1:
    st.markdown("<h2 style='color: #1B4332; font-weight: 800; margin-bottom: 2px;'>📊 Farm History & Outbreak Analytics</h2>", unsafe_allow_html=True)
    st.write(f"Showing historical diagnostics for **{u.get('farm_name', u['full_name'])}** (Operator: **{u['full_name']}**)")
with c_h2:
    if st.button("← Return to Overview", use_container_width=True):
        st.switch_page("app.py")

st.markdown("---")

scans = get_user_scans(u["id"])

if not scans:
    st.info("No field scans recorded yet. Run a diagnostic scan in **Field Diagnostics** to start tracking.")
else:
    df = pd.DataFrame(scans)

    # Overview Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-card"><div class="label">Total Scans Recorded</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="label">Critical Outbreaks</div><div class="value" style="color: #9D0208;">{len(df[df["severity"] == "Critical"])}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="label">Healthy Inspections</div><div class="value" style="color: #2D6A4F;">{len(df[df["severity"] == "Healthy"])}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Pathology Distribution")
    chart_df = df["primary_condition"].value_counts().reset_index()
    chart_df.columns = ["Diagnosis", "Count"]
    chart_df["Diagnosis"] = chart_df["Diagnosis"].str.replace("_", " ").str.title()
    st.bar_chart(chart_df.set_index("Diagnosis"), color="#2D6A4F")

    st.markdown("---")
    st.markdown("### 📋 Complete Farm Audit Log")
    display_df = df[["created_at", "plot_id", "crop_species", "primary_condition", "severity", "confidence", "hotspots_count"]].copy()
    display_df.columns = ["Timestamp", "Plot ID", "Crop", "Diagnosis", "Severity", "Certainty", "Hotspots"]
    display_df["Diagnosis"] = display_df["Diagnosis"].str.replace("_", " ").str.title()
    display_df["Certainty"] = display_df["Certainty"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(display_df, use_container_width=True)