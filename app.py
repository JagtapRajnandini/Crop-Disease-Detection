"""
app.py
------
AgriScan AI — Modern Commercial Landing Page & Authentication Gateway
Rich Emerald & Forest Green Aesthetic
"""

import streamlit as st
from src.auth.database import register_user, login_user

st.set_page_config(
    page_title="AgriScan AI — Precision Agricultural Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None  # None, 'login', or 'register'

# ── Rich Emerald & Forest Green Design System ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F3F8F4; color: #0F2D1E; }

    /* Top Navigation Header */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 28px;
        background: #FFFFFF;
        border-radius: 18px;
        border: 1px solid #D8EAD9;
        box-shadow: 0 4px 20px rgba(27, 67, 50, 0.05);
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1B4332;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.5px;
    }
    .brand-badge {
        background: #E8F5E9;
        color: #2D6A4F;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        border: 1px solid #C8E6C9;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0B291A 0%, #1B4332 50%, #2D6A4F 100%);
        border-radius: 24px;
        padding: 56px 48px;
        color: white;
        box-shadow: 0 16px 40px rgba(11, 41, 26, 0.22);
        margin-bottom: 32px;
        border: 1px solid rgba(216, 243, 220, 0.2);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(149, 213, 178, 0.45);
        color: #D8F3DC;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.18;
        color: #FFFFFF !important;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #B7E4C7;
        max-width: 760px;
        line-height: 1.6;
        margin-bottom: 28px;
    }

    /* Primary Buttons Styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #40916C !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(27, 67, 50, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #40916C 0%, #2D6A4F 100%) !important;
        box-shadow: 0 6px 20px rgba(45, 106, 79, 0.3) !important;
    }

    /* Stats Card Strip */
    .stats-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #D8EAD9;
        border-top: 4px solid #2D6A4F;
        box-shadow: 0 6px 18px rgba(27, 67, 50, 0.04);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stats-card:hover { transform: translateY(-4px); }
    .stats-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B4332;
        margin-bottom: 2px;
    }
    .stats-label {
        font-size: 0.8rem;
        color: #40916C;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Feature Box Grid */
    .feature-box {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 32px 26px;
        border: 1px solid #D8EAD9;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.04);
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-box:hover {
        border-color: #52B788;
        box-shadow: 0 12px 30px rgba(45, 106, 79, 0.12);
        transform: translateY(-4px);
    }
    .feature-icon-wrapper {
        width: 58px;
        height: 58px;
        border-radius: 16px;
        background: #E8F5E9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 20px;
        border: 1px solid #C8E6C9;
    }
    .feature-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #1B4332;
        margin-bottom: 10px;
    }
    .feature-desc {
        font-size: 0.92rem;
        color: #4A6B5B;
        line-height: 1.6;
    }

    /* Auth Modal Container */
    .auth-container {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 36px 40px;
        border: 2px solid #95D5B2;
        box-shadow: 0 16px 44px rgba(27, 67, 50, 0.14);
        max-width: 520px;
        margin: 0 auto 36px auto;
    }
</style>
""", unsafe_allow_html=True)

# ── 1. Top Navbar Header ───────────────────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([4, 2, 2])

with nav_col1:
    st.markdown("""
    <div class="brand-title">
        🌿 AgriScan AI 
        <span class="brand-badge">SaaS Field Core</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    if st.session_state.user:
        u = st.session_state.user
        st.write(f"👨‍🌾 **{u['full_name']}** ({u.get('farm_name', 'My Farm')})")
    else:
        st.write("")

with nav_col3:
    if st.session_state.user:
        if st.button("🚪 Sign Out", key="top_logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.auth_mode = None
            st.rerun()
    else:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Sign In", key="top_login_btn", use_container_width=True):
                st.session_state.auth_mode = "login"
        with btn_c2:
            if st.button("Register", key="top_reg_btn", type="primary", use_container_width=True):
                st.session_state.auth_mode = "register"

# ── 2. Modal Drawer for Authentication ─────────────────────────────────────────
if st.session_state.auth_mode and not st.session_state.user:
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    if st.session_state.auth_mode == "login":
        st.markdown("<h3 style='color: #1B4332; margin-bottom: 2px;'>🔐 Farmer Portal Sign In</h3>", unsafe_allow_html=True)
        st.caption("Access your private farm scan history and longitudinal outbreak trends.")
        
        login_u = st.text_input("Username", key="modal_login_u")
        login_p = st.text_input("Password", type="password", key="modal_login_p")
        
        c_sub, c_cancel = st.columns(2)
        with c_sub:
            if st.button("Sign In", type="primary", use_container_width=True):
                user_obj, msg = login_user(login_u, login_p)
                if user_obj:
                    st.session_state.user = user_obj
                    st.session_state.auth_mode = None
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with c_cancel:
            if st.button("Close", key="cancel_login", use_container_width=True):
                st.session_state.auth_mode = None
                st.rerun()

    elif st.session_state.auth_mode == "register":
        st.markdown("<h3 style='color: #1B4332; margin-bottom: 2px;'>🌱 Create Farmer Account</h3>", unsafe_allow_html=True)
        st.caption("Register your greenhouse or field to begin tracking foliar diseases.")
        
        reg_name = st.text_input("Full Name", key="modal_r_name")
        reg_farm = st.text_input("Farm / Greenhouse Name", key="modal_r_farm")
        reg_loc  = st.text_input("District / Location", key="modal_r_loc")
        reg_user = st.text_input("Choose Username", key="modal_r_user")
        reg_pwd  = st.text_input("Choose Password", type="password", key="modal_r_pwd")
        
        c_sub, c_cancel = st.columns(2)
        with c_sub:
            if st.button("Complete Registration", type="primary", use_container_width=True):
                ok, msg = register_user(reg_user, reg_pwd, reg_name, reg_farm, reg_loc)
                if ok:
                    st.success(msg)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(msg)
        with c_cancel:
            if st.button("Close", key="cancel_reg", use_container_width=True):
                st.session_state.auth_mode = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── 3. Hero Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">⚡ Two-Stage Cascaded Vision · In-The-Wild Field Diagnostics</div>
    <div class="hero-title">Precision Crop Disease Detection & Field Surveillance Station</div>
    <div class="hero-subtitle">
        Eliminate cross-species misdiagnoses with cascaded deep learning. Localize foliar disease hotspots, identify botanical hosts, and generate actionable treatment prescriptions in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# ── 4. Call to Action Navigation (Auth-Aware) ───────────────────────────────────
if st.session_state.user:
    cta_c1, cta_c2, cta_c3 = st.columns([2.5, 2.5, 3])
    with cta_c1:
        if st.button("🔬 Launch Field Diagnostics →", type="primary", use_container_width=True):
            st.switch_page("pages/Field_Diagnostics.py")
    with cta_c2:
        if st.button("📊 View Farm Audit History →", use_container_width=True):
            st.switch_page("pages/Farm_History.py")
else:
    cta_c1, cta_c2 = st.columns([3, 5])
    with cta_c1:
        if st.button("🔬 Try Field Diagnostics (Guest) →", type="primary", use_container_width=True):
            st.switch_page("pages/Field_Diagnostics.py")
    with cta_c2:
        st.markdown(
            "<div style='padding: 10px 16px; background: #E8F5E9; border-radius: 12px; border: 1px solid #C8E6C9; color: #2D6A4F; font-size: 0.88rem; font-weight: 600;'>"
            "💡 <b>Guest Mode Active:</b> You can run live disease scans freely. Sign in above to unlock your permanent <b>Farm Audit History</b>."
            "</div>",
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── 5. Key Performance Metrics Strip ───────────────────────────────────────────
stat1, stat2, stat3, stat4 = st.columns(4)
with stat1:
    st.markdown('<div class="stats-card"><div class="stats-number">29</div><div class="stats-label">Disease Pathologies</div></div>', unsafe_allow_html=True)
with stat2:
    st.markdown('<div class="stats-card"><div class="stats-number">13</div><div class="stats-label">Botanical Crops</div></div>', unsafe_allow_html=True)
with stat3:
    st.markdown('<div class="stats-card"><div class="stats-number">57.5%</div><div class="stats-label">Detector mAP@0.50</div></div>', unsafe_allow_html=True)
with stat4:
    st.markdown('<div class="stats-card"><div class="stats-number">&lt; 100ms</div><div class="stats-label">Inference Latency</div></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── 6. Architectural Advantages & Core Features ────────────────────────────────
st.markdown("<h2 style='color: #1B4332; font-weight: 800; margin-bottom: 24px;'>Engineered for Field-Ready Performance</h2>", unsafe_allow_html=True)

feat1, feat2, feat3 = st.columns(3)

with feat1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon-wrapper">🧠</div>
        <div class="feature-title">Two-Stage Gated Vision</div>
        <div class="feature-desc">
            Standard single-stage detectors confuse similar foliage (e.g. Potato vs Tomato). Our ResNet18 backbone validates the plant family before YOLOv8 localizes lesions.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon-wrapper">🛡️</div>
        <div class="feature-title">Agronomic Advisory Core</div>
        <div class="feature-desc">
            Diagnostics without solutions are useless. AgriScan AI maps 18 disease classes to root causes, moisture triggers, cultural controls, and chemical/organic spray recipes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat3:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon-wrapper">📈</div>
        <div class="feature-title">Farm Outbreak Tracking</div>
        <div class="feature-desc">
            Log scans by greenhouse bed and plot ID. Visualize disease distribution over time and export official timestamped diagnostic summaries for agricultural insurance or agronomists.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── 7. Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #52796F; font-size: 0.85rem; padding: 24px 0; border-top: 1px solid #D8EAD9;">
    AgriScan AI · MKSSS Cummins College of Engineering for Women, Pune · Department of Computer Engineering
</div>
""", unsafe_allow_html=True)