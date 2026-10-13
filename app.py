import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import json
import datetime
from pathlib import Path

# ── Import AI Prediction Engine ────────────────────────────────────────────────
try:
    from src.inference.predict_two_stage import predict_image
    TWO_STAGE_AVAILABLE = True
except ImportError:
    try:
        from src.inference.predict import predict_image
    except ImportError:
        predict_image = None
    TWO_STAGE_AVAILABLE = False

# ── Load Agronomic Advisory Knowledge Base ─────────────────────────────────────
ADVISORY_PATH = Path("src/data/disease_advisory.json")
ADVISORY_DATA = {}
if ADVISORY_PATH.exists():
    try:
        with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
            ADVISORY_DATA = json.load(f)
    except Exception:
        pass

# ── All 13 Supported Species ───────────────────────────────────────────────────
ALL_CROPS = [
    "Auto-Detect via Field Vision",
    "Apple", "Bell Pepper", "Blueberry", "Cherry",
    "Corn", "Grape", "Peach", "Potato", "Raspberry",
    "Soyabean", "Squash", "Strawberry", "Tomato",
]

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriScan AI — Field Diagnostics Station",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Background ── */
    .stApp { background-color: #F4F9F4; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1B4332 0%, #2D6A4F 100%);
    }
    [data-testid="stSidebar"] * { color: #D8F3DC !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #95D5B2 !important; }
    [data-testid="stSidebar"] .stSlider label { color: #B7E4C7 !important; }
    [data-testid="stSidebar"] hr { border-color: #40916C; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 8px 32px rgba(27,67,50,0.18);
    }
    .hero-banner h1 {
        color: #D8F3DC !important;
        font-size: 2.0rem;
        margin-bottom: 4px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #95D5B2;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }
    .status-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(149,213,178,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #D8F3DC;
        margin-right: 8px;
        margin-top: 4px;
    }

    /* ── Intake Form Card ── */
    .intake-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 22px 28px;
        border: 1px solid #D8F3DC;
        box-shadow: 0 2px 12px rgba(45,106,79,0.08);
        margin-bottom: 20px;
    }
    .intake-card h4 {
        color: #1B4332;
        margin-bottom: 4px;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .intake-card p {
        color: #52796F;
        font-size: 0.83rem;
        margin-bottom: 0;
    }

    /* ── Summary Metric Cards ── */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid #2D6A4F;
        box-shadow: 0 2px 8px rgba(45,106,79,0.08);
        height: 100%;
    }
    .metric-card .label {
        color: #52796F;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .metric-card .value {
        color: #1B4332;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* ── Severity Badges ── */
    .severity-critical {
        background: #FFF0F0; border: 1.5px solid #E63946;
        color: #9D0208; border-radius: 8px; padding: 8px 16px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }
    .severity-moderate {
        background: #FFFBEA; border: 1.5px solid #D4A017;
        color: #7B5E00; border-radius: 8px; padding: 8px 16px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }
    .severity-low {
        background: #F0FFF4; border: 1.5px solid #40916C;
        color: #1B4332; border-radius: 8px; padding: 8px 16px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }

    /* ── Advisory Cards ── */
    .adv-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #D8F3DC;
        box-shadow: 0 2px 8px rgba(45,106,79,0.07);
        height: 100%;
    }
    .adv-card h4 { color: #1B4332; font-size: 0.95rem; font-weight: 700; margin-bottom: 10px; }
    .adv-card p  { color: #354F52; font-size: 0.88rem; line-height: 1.6; }

    /* ── Override Warning ── */
    .override-badge {
        background: #FFF8E1; border: 1px solid #FFC107;
        border-radius: 8px; padding: 8px 16px;
        font-size: 0.85rem; color: #856404;
        margin-bottom: 12px; display: block;
    }

    /* ── Section Headers ── */
    .section-header {
        color: #1B4332;
        font-size: 1.1rem;
        font-weight: 700;
        border-bottom: 2px solid #D8F3DC;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #E9F5EE;
        border-radius: 8px 8px 0 0;
        color: #2D6A4F;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #2D6A4F !important;
        color: white !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── General heading color ── */
    h1, h2, h3 { color: #1B4332; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AgriScan AI")
    st.markdown("*Field Diagnostics & Pathology Station*")
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.10, max_value=0.90, value=0.20, step=0.05,
        help="Lower = detects subtle early-stage lesions. Higher = only high-certainty finds."
    )

    use_two_stage = False
    if TWO_STAGE_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🧠 Vision Core Mode")
        use_two_stage = st.toggle(
            "Two-Stage Deep Vision Pipeline",
            value=True,
            help="Stage 1: ResNet18 identifies crop species. Stage 2: YOLOv8 localizes disease regions. Eliminates cross-species misclassification."
        )

    st.markdown("---")
    st.markdown("### 📡 System Status")
    st.markdown(f"{'🟢 Two-Stage Vision Core' if TWO_STAGE_AVAILABLE else '🟡 Single-Stage YOLO Only'}")
    st.markdown(f"{'🟢 Advisory Engine Linked' if ADVISORY_DATA else '🟡 Advisory DB Offline'}")
    st.markdown("🟢 13 Species Supported")

    st.markdown("---")
    st.caption("Supported: Apple · Bell Pepper · Blueberry · Cherry · Corn · Grape · Peach · Potato · Raspberry · Soyabean · Squash · Strawberry · Tomato")


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🌿 AgriScan AI — Field Diagnostics & Pathology Station</h1>
    <p>Precision foliar disease surveillance powered by two-stage deep vision. Upload a field photograph for instant pathogen classification, infection localization, and agronomic treatment prescription.</p>
    <span class="status-badge">● Vision Core Active</span>
    <span class="status-badge">🌿 13 Species Supported</span>
    <span class="status-badge">🛡️ Advisory Engine Linked</span>
    <span class="status-badge">📍 PlantDoc · 29 Disease Classes</span>
</div>
""", unsafe_allow_html=True)


# ── Farmer Intake Form ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Field Intake Form</div>', unsafe_allow_html=True)

intake_col1, intake_col2, intake_col3 = st.columns([2, 2, 3])

with intake_col1:
    manual_crop = st.selectbox(
        "🌾 Target Crop / Plot Type",
        options=ALL_CROPS,
        index=0,
        help="Select 'Auto-Detect via Field Vision' to let AI identify the species, or manually select your crop to force species-specific filtering."
    )

with intake_col2:
    plot_id = st.text_input(
        "📍 Field / Plot Identifier (Optional)",
        placeholder="e.g. North Greenhouse – Bed 4",
        help="Label this scan for your farm records."
    )

with intake_col3:
    uploaded_file = st.file_uploader(
        "📷 Upload Field Photograph (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an in-the-wild crop leaf photograph for analysis."
    )

st.markdown("---")

# ── Main Analysis ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Override badge
    if manual_crop != "Auto-Detect via Field Vision":
        st.markdown(
            f'<span class="override-badge">⚠️ <b>Manual Crop Override Active:</b> Analysis locked to <b>{manual_crop}</b>. ResNet auto-detection bypassed.</span>',
            unsafe_allow_html=True
        )

    # ── Run Inference ──────────────────────────────────────────────────────────
    with st.spinner("🔬 Running Two-Stage Field Diagnostics... Please wait."):
        try:
            crop_override = manual_crop if manual_crop != "Auto-Detect via Field Vision" else None
            if TWO_STAGE_AVAILABLE:
                result = predict_image(
                    image,
                    conf_threshold=conf_threshold,
                    use_two_stage=use_two_stage,
                    manual_crop=crop_override
                )
            elif predict_image:
                result = predict_image(image, conf_threshold=conf_threshold)
            else:
                st.error("No inference engine found. Check your src/inference directory.")
                result = None
        except Exception as e:
            st.error(f"❌ Inference Error: {e}")
            result = None

    if result:
        # ── Severity Calculation ───────────────────────────────────────────────
        top_conf = result["highest_confidence"]
        num_regions = result["total_regions_detected"]
        is_healthy = "healthy" in result["primary_condition"].lower() or num_regions == 0

        if is_healthy:
            severity_label = "🟢 Healthy / No Pathogen Detected"
            severity_class = "severity-low"
        elif top_conf >= 0.70 or num_regions >= 3:
            severity_label = "🔴 Critical — Immediate Intervention Required"
            severity_class = "severity-critical"
        elif top_conf >= 0.45 or num_regions >= 1:
            severity_label = "🟡 Moderate — Prompt Attention Advised"
            severity_class = "severity-moderate"
        else:
            severity_label = "🟢 Low / Negligible — Monitor Field"
            severity_class = "severity-low"

        # ── Summary Header ─────────────────────────────────────────────────────
        st.markdown(f'<div class="section-header">🧬 Diagnostic Report — {plot_id if plot_id else "Field Specimen"}</div>', unsafe_allow_html=True)

        # Species confidence info
        if result.get("manual_override"):
            st.info(f"🌾 **Crop Override Active:** Locked to **{result['primary_plant']}** — ResNet bypassed.")
        elif use_two_stage and result.get("species_confidence", 0) > 0:
            conf_pct = result["species_confidence"] * 100
            badge = "✅" if conf_pct >= 75 else "⚠️"
            filter_status = "Species filter active" if conf_pct >= 75 else "Low confidence — filter inactive, all YOLO detections shown"
            st.info(f"{badge} **AI Species Identification Certainty:** {conf_pct:.1f}% — *{filter_status}*")

        # Severity badge
        st.markdown(f'<div class="{severity_class}">{severity_label}</div><br>', unsafe_allow_html=True)

        # ── Metric Cards ───────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        cards = [
            ("Host Species", result["primary_plant"]),
            ("Pathogen Classification", result["primary_condition"].replace("_", " ").title()),
            ("Diagnostic Certainty", f"{top_conf * 100:.1f}%"),
            ("Infection Hotspots", f"{num_regions} Region(s)"),
        ]
        for col, (label, value) in zip([m1, m2, m3, m4], cards):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ───────────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🔍 Visual Inspection",
            "💊 Agronomic Advisory",
            "📄 Export & Log"
        ])

        # ── TAB 1: Visual Inspection ───────────────────────────────────────────
        with tab1:
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("**📷 Original Field Photograph**")
                st.image(image, use_container_width=True)
            with img_col2:
                st.markdown("**🔬 AI Disease Localization — Infection Hotspot Map**")
                st.image(result["annotated_image"], use_container_width=True)

            if result["detections"]:
                st.markdown("---")
                st.markdown("**📊 Detailed Pathological Breakdown**")
                df = pd.DataFrame(result["detections"])
                df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.1f}%")
                df["class"] = df["class"].str.replace("_", " ").str.title()
                df = df.rename(columns={
                    "class": "Disease Class",
                    "plant": "Host Plant",
                    "confidence": "Diagnostic Certainty"
                })
                if "box" in df.columns:
                    df = df.drop(columns=["box"])
                st.dataframe(df, use_container_width=True)
            else:
                st.success("✅ No disease hotspots localized. Canopy appears healthy.")

        # ── TAB 2: Agronomic Advisory ──────────────────────────────────────────
        with tab2:
            if is_healthy:
                st.success(
                    "### 🟢 Foliar Health Status: Healthy Canopy\n\n"
                    "No active fungal, bacterial, or viral pathogens were localized on this specimen. "
                    "Continue regular field scouting, balanced N-P-K nutrient management, and maintain "
                    "drip irrigation schedules. Schedule next scan in 7–14 days."
                )
            elif result["detections"]:
                # Lookup advisory for top detection
                raw_conf = lambda d: float(d["confidence"].replace("%", "")) if isinstance(d["confidence"], str) else d["confidence"]
                top_det = max(result["detections"], key=raw_conf)
                top_class = top_det.get("class", "").lower().replace(" ", "_")

                advisory = ADVISORY_DATA.get(top_class)
                if not advisory:
                    for k, v in ADVISORY_DATA.items():
                        if k in top_class or top_class in k:
                            advisory = v
                            break

                if advisory:
                    st.markdown(f"### 🛡️ Agronomic Management Prescription: **{advisory['disease_name']}**")
                    st.caption(f"**Identified Pathogen / Causal Agent:** {advisory['pathogen']}")
                    st.markdown("<br>", unsafe_allow_html=True)

                    adv_c1, adv_c2, adv_c3 = st.columns(3)
                    with adv_c1:
                        st.markdown(
                            f'<div class="adv-card">'
                            f'<h4>🔍 Root Causes & Environmental Triggers</h4>'
                            f'<p>{advisory["causes"].replace(chr(10), "<br>")}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with adv_c2:
                        st.markdown(
                            f'<div class="adv-card">'
                            f'<h4>🛡️ Cultural & Preventative Controls</h4>'
                            f'<p>{advisory["prevention"].replace(chr(10), "<br>")}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with adv_c3:
                        st.markdown(
                            f'<div class="adv-card">'
                            f'<h4>💊 Chemical / Organic Spray Prescription</h4>'
                            f'<p>{advisory["treatment"].replace(chr(10), "<br>")}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info(
                        f"🔬 **Detected:** `{top_class.replace('_', ' ').title()}` — "
                        f"Advisory data not available in current database. "
                        f"Consult your local agricultural extension office or Krishi Vigyan Kendra."
                    )
            else:
                st.info("Run a scan first to view agronomic advisory recommendations.")

        # ── TAB 3: Export & Log ────────────────────────────────────────────────
        with tab3:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            plot_label = plot_id if plot_id else "Unspecified Field"

            st.markdown("### 📄 Diagnostic Field Summary")
            st.caption(f"Generated: {timestamp}")

            # ── Visual Field Report Card ───────────────────────────────────────
            crop_mode_str = f"Manual Override — {manual_crop}" if result.get("manual_override") else "AI Auto-Detected (ResNet18)"
            crop_conf_str = "100% (Manual)" if result.get("manual_override") else f"{result.get('species_confidence', 0) * 100:.1f}%"
            pathogen_title = result["primary_condition"].replace("_", " ").title()

            report_card_html = f"""<div style="background: #FFFFFF; border-radius: 14px; border: 1px solid #D8F3DC; box-shadow: 0 4px 16px rgba(27,67,50,0.08); overflow: hidden; margin-bottom: 20px;">
<div style="background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%); padding: 20px 28px; color: white;">
<div style="font-size: 0.78rem; color: #95D5B2; text-transform: uppercase; letter-spacing: 1px;">AgriScan AI · Field Diagnostics Report</div>
<div style="font-size: 1.25rem; font-weight: 800; margin-top: 4px; color: #D8F3DC;">{plot_label}</div>
<div style="font-size: 0.82rem; color: #95D5B2; margin-top: 2px;">📅 {timestamp}</div>
</div>
<div style="display: flex; border-bottom: 1px solid #E9F5EE;">
<div style="flex:1; padding: 16px 20px; border-right: 1px solid #E9F5EE;">
<div style="font-size: 0.72rem; color: #52796F; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Host Species</div>
<div style="font-size: 1.1rem; font-weight: 800; color: #1B4332; margin-top: 2px;">{result['primary_plant']}</div>
</div>
<div style="flex:1; padding: 16px 20px; border-right: 1px solid #E9F5EE;">
<div style="font-size: 0.72rem; color: #52796F; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Pathogen</div>
<div style="font-size: 1.1rem; font-weight: 800; color: #1B4332; margin-top: 2px;">{pathogen_title}</div>
</div>
<div style="flex:1; padding: 16px 20px; border-right: 1px solid #E9F5EE;">
<div style="font-size: 0.72rem; color: #52796F; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Certainty</div>
<div style="font-size: 1.1rem; font-weight: 800; color: #1B4332; margin-top: 2px;">{top_conf * 100:.1f}%</div>
</div>
<div style="flex:1; padding: 16px 20px;">
<div style="font-size: 0.72rem; color: #52796F; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Hotspots</div>
<div style="font-size: 1.1rem; font-weight: 800; color: #1B4332; margin-top: 2px;">{num_regions} Region(s)</div>
</div>
</div>
<div style="padding: 14px 24px; background: #F4F9F4; border-bottom: 1px solid #E9F5EE;">
<span style="font-size: 0.78rem; color: #52796F; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">Severity Assessment → </span>
<span style="font-weight: 700; color: #1B4332;">{severity_label}</span>
</div>
<div style="padding: 14px 24px; display: flex; gap: 40px;">
<div>
<div style="font-size: 0.72rem; color: #52796F; font-weight: 600; text-transform: uppercase;">Identification Mode</div>
<div style="font-size: 0.9rem; color: #1B4332; font-weight: 700;">{crop_mode_str}</div>
</div>
<div>
<div style="font-size: 0.72rem; color: #52796F; font-weight: 600; text-transform: uppercase;">Species Confidence</div>
<div style="font-size: 0.9rem; color: #1B4332; font-weight: 700;">{crop_conf_str}</div>
</div>
</div>
</div>"""
            st.markdown(report_card_html, unsafe_allow_html=True)

            # ── Infection Hotspots List ────────────────────────────────────────
            if result["detections"]:
                st.markdown("**🦠 Detected Infection Regions**")
                for i, d in enumerate(result["detections"], 1):
                    conf_val = d['confidence'] if isinstance(d['confidence'], float) else float(d['confidence'].replace('%', '')) / 100
                    bar_color = "#E63946" if conf_val >= 0.70 else ("#D4A017" if conf_val >= 0.45 else "#40916C")
                    bar_width = max(int(conf_val * 100), 5)
                    cls_name = d['class'].replace('_', ' ').title()

                    bar_html = f"""<div style="background: #FFFFFF; border-radius: 10px; border: 1px solid #E9F5EE; padding: 12px 18px; margin-bottom: 8px; display: flex; align-items: center; gap: 16px;">
<div style="background: #F4F9F4; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #2D6A4F; font-size: 0.85rem; flex-shrink: 0;">{i}</div>
<div style="flex: 1;">
<div style="font-weight: 700; color: #1B4332; font-size: 0.92rem;">{cls_name}</div>
<div style="margin-top: 6px; background: #F4F9F4; border-radius: 4px; height: 6px; overflow: hidden;">
<div style="width: {bar_width}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
</div>
</div>
<div style="font-weight: 800; font-size: 1rem; color: {bar_color}; flex-shrink: 0; min-width: 50px; text-align: right;">{conf_val * 100:.1f}%</div>
</div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Treatment Summary Snippet ──────────────────────────────────────
            if ADVISORY_DATA and result["detections"]:
                raw_conf_fn = lambda d: d["confidence"] if isinstance(d["confidence"], float) else float(d["confidence"].replace("%", "")) / 100
                top_det_exp = max(result["detections"], key=raw_conf_fn)
                top_cls_exp = top_det_exp.get("class", "").lower().replace(" ", "_")
                adv_exp = ADVISORY_DATA.get(top_cls_exp)
                if not adv_exp:
                    for k, v in ADVISORY_DATA.items():
                        if k in top_cls_exp or top_cls_exp in k:
                            adv_exp = v
                            break

                if adv_exp:
                    treat_snip = adv_exp['treatment'][:220] + ("..." if len(adv_exp['treatment']) > 220 else "")
                    adv_snippet_html = f"""<div style="background: #F4F9F4; border-radius: 12px; border: 1px solid #D8F3DC; padding: 18px 22px; margin-bottom: 20px;">
<div style="font-size: 0.78rem; color: #52796F; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 10px;">🛡️ Treatment Prescription Summary — {adv_exp['disease_name']}</div>
<div style="display: flex; gap: 20px;">
<div style="flex:1;">
<div style="font-size: 0.75rem; font-weight: 700; color: #2D6A4F; margin-bottom: 4px;">PATHOGEN</div>
<div style="font-size: 0.88rem; color: #354F52;">{adv_exp['pathogen']}</div>
</div>
<div style="flex:2;">
<div style="font-size: 0.75rem; font-weight: 700; color: #2D6A4F; margin-bottom: 4px;">RECOMMENDED ACTION</div>
<div style="font-size: 0.88rem; color: #354F52;">{treat_snip}</div>
</div>
</div>
</div>"""
                    st.markdown(adv_snippet_html, unsafe_allow_html=True)

            # ── Download Button ────────────────────────────────────────────────
            summary_lines = [
                "=" * 60,
                "     AGRISCAN AI — FIELD DIAGNOSTIC REPORT",
                "=" * 60,
                f"  Date & Time           : {timestamp}",
                f"  Field / Plot ID        : {plot_label}",
                f"  Host Species           : {result['primary_plant']}",
                f"  Crop ID Mode           : {crop_mode_str}",
                "-" * 60,
                f"  PATHOGEN CLASSIFICATION: {pathogen_title}",
                f"  DIAGNOSTIC CERTAINTY   : {top_conf * 100:.1f}%",
                f"  INFECTION HOTSPOTS     : {num_regions} Region(s)",
                f"  SEVERITY               : {severity_label}",
                "-" * 60,
            ]
            if result["detections"]:
                summary_lines.append("  DETECTIONS:")
                for i, d in enumerate(result["detections"], 1):
                    cv = d['confidence'] if isinstance(d['confidence'], float) else float(d['confidence'].replace('%', '')) / 100
                    summary_lines.append(f"    [{i}] {d['class'].replace('_', ' ').title()} — {cv * 100:.1f}%")
            summary_lines += ["=" * 60, "  Generated by AgriScan AI — MKSSS Cummins COE", "=" * 60]
            report_text = "\n".join(summary_lines)

            st.download_button(
                label="⬇️ Download Diagnostic Field Summary (.txt)",
                data=report_text,
                file_name=f"AgriScan_{plot_label.replace(' ', '_')}_{file_timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    # ── Landing Placeholder ────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: #FFFFFF;
        border: 2px dashed #95D5B2;
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
        color: #52796F;
    ">
        <div style="font-size: 3.5rem;">🌾</div>
        <h3 style="color: #2D6A4F; margin-top: 12px;">Ready for Field Diagnostics</h3>
        <p style="max-width: 500px; margin: 0 auto; line-height: 1.7;">
            Select your target crop and upload a field photograph above to begin precision disease surveillance.
            The AI pipeline will identify the pathogen, localize infection hotspots, and generate a full agronomic treatment prescription.
        </p>
        <br>
        <p style="font-size: 0.82rem; color: #84A98C;">
            Supports: Apple · Bell Pepper · Blueberry · Cherry · Corn · Grape ·
            Peach · Potato · Raspberry · Soyabean · Squash · Strawberry · Tomato
        </p>
    </div>
    """, unsafe_allow_html=True)