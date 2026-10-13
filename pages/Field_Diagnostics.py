"""
pages/Field_Diagnostics.py
--------------------------
Screen 1: Field Intake, Two-Stage AI Diagnostics, Canopy Health Score & Advisory
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import json
import datetime
from pathlib import Path

from src.auth.database import save_scan

# Import AI Engine
try:
    from src.inference.predict_two_stage import predict_image
    TWO_STAGE_AVAILABLE = True
except ImportError:
    try:
        from src.inference.predict import predict_image
    except ImportError:
        predict_image = None
    TWO_STAGE_AVAILABLE = False

ADVISORY_PATH = Path("src/data/disease_advisory.json")
ADVISORY_DATA = {}
if ADVISORY_PATH.exists():
    try:
        with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
            ADVISORY_DATA = json.load(f)
    except Exception:
        pass

ALL_CROPS = [
    "Auto-Detect via Field Vision",
    "Apple", "Bell Pepper", "Blueberry", "Cherry",
    "Corn", "Grape", "Peach", "Potato", "Raspberry",
    "Soyabean", "Squash", "Strawberry", "Tomato",
]

DISEASE_KEYWORDS = [
    "blight", "spot", "rot", "rust", "scab", "virus", "mold",
    "mildew", "speck", "canker", "mosaic", "yellow"
]

def compute_canopy_health_score(condition_str, detections, top_confidence):
    """
    Computes a realistic Canopy Health Index (0 - 100%):
    - Confirmed healthy foliage: 95 - 100%
    - 1 minor spot: 75 - 85%
    - Moderate disease (1-2 regions): 50 - 70%
    - Severe / multiple lesions: 15 - 45%
    """
    if not detections:
        return 96, "OPTIMAL", "#2D6A4F"

    has_disease = any(
        any(k in d["class"].lower() for k in DISEASE_KEYWORDS)
        for d in detections
    )

    if not has_disease:
        return 98, "OPTIMAL", "#2D6A4F"

    penalty = 0
    num_spots = len(detections)

    penalty += min(num_spots * 15, 45)
    penalty += int(top_confidence * 30)

    cond_lower = condition_str.lower()
    if any(aggr in cond_lower for aggr in ["blight", "rot", "rust", "virus"]):
        penalty += 15

    health_score = max(100 - penalty, 12)

    if health_score >= 80:
        return health_score, "FAIR", "#40916C"
    elif health_score >= 55:
        return health_score, "VULNERABLE", "#D4A017"
    else:
        return health_score, "CRITICAL", "#E63946"

st.set_page_config(page_title="Field Diagnostics — AgriScan AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAF8; color: #1B382B; }

    /* Fixed Height & Flex Alignment for Uniform KPI Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 16px 18px;
        border-left: 5px solid #2D6A4F;
        box-shadow: 0 4px 12px rgba(27,67,50,0.04);
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card .label {
        color: #52796F;
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #1B4332;
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 4px;
    }
    .tier-badge {
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 700;
        vertical-align: middle;
        margin-left: 4px;
        display: inline-block;
    }
    
    .severity-critical { background: #FFF0F0; border: 1.5px solid #E63946; color: #9D0208; border-radius: 8px; padding: 8px 16px; font-weight: 700; display: inline-block; }
    .severity-moderate { background: #FFFBEA; border: 1.5px solid #D4A017; color: #7B5E00; border-radius: 8px; padding: 8px 16px; font-weight: 700; display: inline-block; }
    .severity-low { background: #F0FFF4; border: 1.5px solid #40916C; color: #1B4332; border-radius: 8px; padding: 8px 16px; font-weight: 700; display: inline-block; }

    .adv-card { background: #FFFFFF; border-radius: 14px; padding: 20px; border: 1px solid #D8F3DC; box-shadow: 0 4px 12px rgba(27,67,50,0.04); height: 100%; }
    .adv-card h4 { color: #1B4332; font-size: 0.95rem; font-weight: 700; margin-bottom: 10px; }
    .adv-card p  { color: #354F52; font-size: 0.88rem; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)

# ── Top Bar ────────────────────────────────────────────────────────────────────
c_h1, c_h2 = st.columns([6, 2])
with c_h1:
    st.markdown("<h2 style='color: #1B4332; font-weight: 800; margin-bottom: 2px;'>🔬 Foliar Pathology & Field Diagnostics</h2>", unsafe_allow_html=True)
    st.caption("Upload crop leaf photographs for two-stage detection, localization, and agronomic prescriptions.")
with c_h2:
    if st.button("← Return to Overview", use_container_width=True):
        st.switch_page("app.py")

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Vision Core Settings")
    conf_threshold = st.slider(
        "Detection Sensitivity", 0.05, 0.90, 0.15, 0.05,
        help="Lower slider to 0.10 to capture subtle or early-stage lesions."
    )
    use_two_stage = st.toggle("Two-Stage Deep Vision", value=True) if TWO_STAGE_AVAILABLE else False
    st.markdown("---")
    if "user" in st.session_state and st.session_state.user:
        u = st.session_state.user
        st.success(f"👨‍🌾 Logged in: **{u['full_name']}**")
    else:
        st.info("Operating in Guest Mode")

# ── Intake Form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    manual_crop = st.selectbox("🌾 Target Crop / Plot Type", ALL_CROPS, index=0)
with col2:
    plot_id = st.text_input("📍 Field / Plot Identifier", placeholder="e.g. Polyhouse 2 – Bed 4")
with col3:
    uploaded_file = st.file_uploader("📷 Upload Leaf Photograph", type=["jpg", "jpeg", "png"])

st.markdown("---")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    crop_override = manual_crop if manual_crop != "Auto-Detect via Field Vision" else None

    with st.spinner("Analyzing foliar specimen with Two-Stage AI pipeline..."):
        try:
            if TWO_STAGE_AVAILABLE:
                result = predict_image(image, conf_threshold=conf_threshold, use_two_stage=use_two_stage, manual_crop=crop_override)
            else:
                result = predict_image(image, conf_threshold=conf_threshold)
        except Exception as e:
            st.error(f"Inference error: {e}")
            result = None

    if result:
        top_conf = result["highest_confidence"]
        num_regions = result["total_regions_detected"]
        detections = result.get("detections", [])

        # ── 1. Calculate Health Score & Classification ─────────────────────────
        health_score, health_tier, health_color = compute_canopy_health_score(
            result["primary_condition"], detections, top_conf
        )

        is_healthy = health_tier == "OPTIMAL"

        if is_healthy:
            severity_label = "🟢 Healthy / No Pathogen Detected"
            sev_class, sev_clean = "severity-low", "Healthy"
            diagnosis_clean = "Healthy Foliage"
        elif health_tier == "CRITICAL":
            severity_label = "🔴 Critical — Immediate Intervention Required"
            sev_class, sev_clean = "severity-critical", "Critical"
            diagnosis_clean = result["primary_condition"].replace("_", " ").title()
        elif health_tier == "VULNERABLE":
            severity_label = "🟡 Moderate — Prompt Attention Advised"
            sev_class, sev_clean = "severity-moderate", "Moderate"
            diagnosis_clean = result["primary_condition"].replace("_", " ").title()
        else:
            severity_label = "🟢 Minor Infection — Regular Field Scouting"
            sev_class, sev_clean = "severity-low", "Low"
            diagnosis_clean = result["primary_condition"].replace("_", " ").title()

        # ── 2. Auto-save scan (If logged in) ───────────────────────────────────
        if "user" in st.session_state and st.session_state.user:
            saved_id = save_scan(
                user_id=st.session_state.user["id"],
                plot_id=plot_id,
                crop_species=result["primary_plant"],
                condition=diagnosis_clean,
                confidence=top_conf if not is_healthy else 0.95,
                severity=sev_clean,
                hotspots=0 if is_healthy else num_regions,
                image_np=result["annotated_image"]
            )
            st.toast(f"💾 Saved to farm history! (Record #{saved_id})", icon="✅")
        else:
            st.info("ℹ️ **Guest Scan:** Results shown above. To store this diagnosis in your permanent farm records, sign in from the home page.")

        # Severity Badge
        st.markdown(f'<div class="{sev_class}">{severity_label}</div><br>', unsafe_allow_html=True)

        # ── 3. Aligned KPI Cards ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Host Species</div>
                <div class="value">{result["primary_plant"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            diag_label = "Canopy Health Status" if is_healthy else "Pathogen Classification"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{diag_label}</div>
                <div class="value">{diagnosis_clean}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            badge_bg = "#E8F5E9" if health_tier == "OPTIMAL" else ("#FEF3C7" if health_tier == "VULNERABLE" else "#FEE2E2")
            score_html = f"""
            <div class="metric-card">
                <div class="label">Canopy Health Score</div>
                <div class="value" style="color: {health_color};">
                    {health_score}% 
                    <span class="tier-badge" style="background: {badge_bg}; color: {health_color};">{health_tier}</span>
                </div>
            </div>
            """
            st.markdown(score_html, unsafe_allow_html=True)
            
        with m4:
            hotspot_val = "0 (Clean Leaf)" if is_healthy else f"{num_regions} Region(s)"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Infection Hotspots</div>
                <div class="value">{hotspot_val}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── 4. Health Score Visual Bar ─────────────────────────────────────────
        st.markdown(f"""
        <div style="background: #E2E8F0; border-radius: 8px; height: 10px; width: 100%; margin: 12px 0 24px 0; overflow: hidden;">
            <div style="background: {health_color}; width: {health_score}%; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── 5. Detailed Inspection Tabs ────────────────────────────────────────
        t1, t2, t3 = st.tabs(["🔍 Visual Inspection", "💊 Agronomic Advisory", "📄 Export Summary"])

        with t1:
            c1, c2 = st.columns(2)
            c1.image(image, caption="Field Photograph", use_container_width=True)
            c2.image(result["annotated_image"], caption="Pathology Localization", use_container_width=True)
            
            if is_healthy:
                st.success("✅ **Clean Foliar Surface:** Verified healthy leaf structure with optimal canopy health score.")
            elif detections:
                df = pd.DataFrame(detections)
                df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
                df["class"] = df["class"].str.replace("_", " ").str.title()
                if "box" in df.columns:
                    df = df.drop(columns=["box"])
                st.dataframe(df, use_container_width=True)

        with t2:
            if is_healthy:
                st.success(
                    "### 🟢 Canopy Health Status: Fully Healthy (Optimal)\n\n"
                    f"**Health Index: {health_score}%** — No active pathogens localized on this specimen. "
                    "Maintain current irrigation schedules and conduct regular weekly field scouting."
                )
            elif detections:
                raw_c = lambda d: float(d["confidence"].replace("%","")) if isinstance(d["confidence"], str) else d["confidence"]
                top_det = max(detections, key=raw_c)
                top_cls = top_det.get("class", "").lower().replace(" ", "_")
                adv = ADVISORY_DATA.get(top_cls)
                if not adv:
                    for k, v in ADVISORY_DATA.items():
                        if k in top_cls or top_cls in k:
                            adv = v
                            break
                if adv:
                    st.markdown(f"### 🛡️ Prescription: **{adv['disease_name']}**")
                    a1, a2, a3 = st.columns(3)
                    a1.markdown(f'<div class="adv-card"><h4>🔍 Root Causes</h4><p>{adv["causes"]}</p></div>', unsafe_allow_html=True)
                    a2.markdown(f'<div class="adv-card"><h4>🛡️ Cultural Prevention</h4><p>{adv["prevention"]}</p></div>', unsafe_allow_html=True)
                    a3.markdown(f'<div class="adv-card"><h4>💊 Spray Prescription</h4><p>{adv["treatment"]}</p></div>', unsafe_allow_html=True)
                else:
                    st.info(f"🔬 Identified: `{top_cls.replace('_', ' ').title()}`. Consult your local agricultural extension officer for dosage instructions.")

        with t3:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_text = (
                f"AGRISCAN FIELD REPORT\n"
                f"Date: {ts}\n"
                f"Plot: {plot_id or 'General'}\n"
                f"Host: {result['primary_plant']}\n"
                f"Diagnosis: {diagnosis_clean}\n"
                f"Health Score: {health_score}% ({health_tier})\n"
                f"Severity: {sev_clean}\n"
                f"Hotspots: {0 if is_healthy else num_regions}\n"
            )
            st.download_button("⬇️ Download Field Report (.txt)", report_text, file_name=f"Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt", use_container_width=True)