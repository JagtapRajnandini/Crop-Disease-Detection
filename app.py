import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# Try to use Two-Stage predictor if available
try:
    from src.inference.predict_two_stage import predict_image
    TWO_STAGE_AVAILABLE = True
except ImportError:
    from src.inference.predict import predict_image
    TWO_STAGE_AVAILABLE = False

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CultiKure - AI Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #F8FAFC; }
    .sidebar .sidebar-content { background: #FFFFFF; }
    .stButton>button { width: 100%; border-radius: 5px; }
    h1, h2, h3 { color: #2E7D32; }
    .summary-card {
        background: #F1F8F5;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=70)
    st.title("Settings & Info")
    
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values detect more subtle symptoms; higher values only show high-certainty detections."
    )
    
    use_two_stage = False
    if TWO_STAGE_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🧠 AI Architecture")
        use_two_stage = st.toggle(
            "Use True Two-Stage AI Pipeline",
            value=True,
            help="Stage 1: ResNet IDs crop. Stage 2: YOLO detects disease. Eliminates cross-species misclassification."
        )

    st.markdown("---")
    st.markdown("### 🌾 Supported Crops (13 Species)")
    st.write("Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Peach, Potato, Raspberry, Soyabean, Squash, Strawberry, Tomato")

# ── Main Application ───────────────────────────────────────────────────────────
st.markdown("<h1>🌿 AI-Based Crop Disease Detection & Localization</h1>", unsafe_allow_html=True)
st.write("Upload a real-world plant photograph to detect diseases and locate affected leaf regions using fine-tuned AI.")

uploaded_file = st.file_uploader("Choose a plant/leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. UI Layout for Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Uploaded Image", use_column_width=True)
        
    with col2:
        with st.spinner("Analyzing image using AI pipeline..."):
            try:
                # Run Inference (Automated toggle)
                if TWO_STAGE_AVAILABLE:
                    result = predict_image(image, conf_threshold=conf_threshold, use_two_stage=use_two_stage)
                else:
                    result = predict_image(image, conf_threshold=conf_threshold)
                
                # Show Annotated Image
                st.image(result["annotated_image"], caption="AI Disease Localization", use_column_width=True)
            except Exception as e:
                st.error(f"Error during AI inference: {e}")
                result = None

    # 3. Summary Dashboard
    if result:
        st.markdown("---")
        st.markdown("### 📋 Crop Health Assessment Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f'<div class="summary-card"><small>Detected Plant</small><h3>{result["primary_plant"]}</h3></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="summary-card"><small>Primary Condition</small><h3>{result["primary_condition"].replace("_", " ").title()}</h3></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="summary-card"><small>Highest Confidence</small><h3>{result["highest_confidence"]*100:.1f}%</h3></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="summary-card"><small>Regions Localized</small><h3>{result["total_regions_detected"]} Box(es)</h3></div>', unsafe_allow_html=True)
            
        # 4. Detailed Detections Table
        if result["detections"]:
            st.markdown("<br><b>Detailed Pathogen Detection:</b>", unsafe_allow_html=True)
            df = pd.DataFrame(result["detections"])
            # Format dataframe for display
            df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
            df["class"] = df["class"].str.replace("_", " ").str.title()
            df = df.drop(columns=["box"])  # Hide coordinates for clean UI
            st.dataframe(df, use_container_width=True)