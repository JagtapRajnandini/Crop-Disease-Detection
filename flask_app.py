"""
flask_app.py
------------
AgriScan AI — Precision Agricultural Intelligence Web Application.
Full Flask implementation featuring:
- Commercial Landing Page & Performance Stats
- SQLite Farmer Authentication & Secure Sessions (Login / Register / Logout)
- Hybrid Foliar Pathology & Lesion Diagnosis (ResNet-50 + SVM)
- Automated Farm Scan Audit Trail & Outbreak History
- Interactive Agronomic Prescriptions & Clinical Dossiers
"""

from pathlib import Path
import re
import os
import io
import json
import base64
from datetime import datetime
import urllib.parse
import yaml
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, make_response, send_from_directory
)
from markupsafe import Markup

# ── Project Roots & Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
ADVISORY_JSON_PATH = DATA_DIR / "disease_advisory.json"
ADVISORY_YAML_PATH = DATA_DIR / "disease_info.yaml"

# ── Database & Auth Helpers ────────────────────────────────────────────────────
try:
    from src.auth.database import (
        init_db, login_user, register_user, save_scan, get_user_scans
    )
    init_db()
except Exception as e:
    print(f"[Warning] Database initialization notice: {e}")
    login_user = None
    register_user = None
    save_scan = None
    get_user_scans = None

# ── Inference Engine Loading ───────────────────────────────────────────────────
try:
    from src.inference.predict_two_stage import predict_image
    TWO_STAGE_AVAILABLE = True
    # Pre-warm AI models into RAM at startup for zero-latency uploads
    try:
        from src.models.deep_feature_extractor import get_feature_extractor
        from src.models.svm_classifier import get_svm_classifier
        get_feature_extractor()
        get_svm_classifier()
    except Exception as e:
        print(f"[Notice] Model pre-warm on startup: {e}")
except ImportError:
    predict_image = None
    TWO_STAGE_AVAILABLE = False

# ── Load and Merge Advisory Catalogues (JSON + YAML) ───────────────────────────
ADVISORY_DATA = {}

if ADVISORY_JSON_PATH.exists():
    try:
        with open(ADVISORY_JSON_PATH, "r", encoding="utf-8") as f:
            ADVISORY_DATA.update(json.load(f))
    except Exception as e:
        print(f"[Warning] Error reading {ADVISORY_JSON_PATH}: {e}")

if ADVISORY_YAML_PATH.exists():
    try:
        with open(ADVISORY_YAML_PATH, "r", encoding="utf-8") as f:
            yaml_adv = yaml.safe_load(f) or {}
            for k, v in yaml_adv.items():
                if k not in ADVISORY_DATA:
                    ADVISORY_DATA[k] = {
                        "disease_name": k.replace("_", " ").title(),
                        "causes": v.get("description", ""),
                        "prevention": v.get("prevention", ""),
                        "treatment": "Apply targeted fungicides as prescribed by agricultural extension officers.",
                        "products": v.get("products", [])
                    }
                else:
                    if "products" not in ADVISORY_DATA[k] and "products" in v:
                        ADVISORY_DATA[k]["products"] = v["products"]
                    if not ADVISORY_DATA[k].get("causes") and v.get("description"):
                        ADVISORY_DATA[k]["causes"] = v["description"]
    except Exception as e:
        print(f"[Warning] Error reading {ADVISORY_YAML_PATH}: {e}")

# ── Flask Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB max upload
app.secret_key = os.urandom(24)

# ── Template Filters ───────────────────────────────────────────────────────────
@app.template_filter("nl2br")
def nl2br_filter(s):
    if not s:
        return ""
    lines = str(s).splitlines()
    escaped = [Markup.escape(line) for line in lines]
    return Markup("<br />").join(escaped)

# ── Health Scoring Logic (Canopy Health Index & Foliar Severity) ──────────────
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

# ── Helper: Image to Base64 String ─────────────────────────────────────────────
def image_to_base64(img_input):
    if isinstance(img_input, np.ndarray):
        img_pil = Image.fromarray(img_input)
    elif isinstance(img_input, Image.Image):
        img_pil = img_input
    else:
        return ""

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ── Helper: Clean Product Formulations & Live Links ────────────────────────────
def clean_product_entry(prod):
    """
    Sanitizes product name and ensures all links are active Google search queries
    for genuine agricultural fungicides and formulations (never broken example.com links).
    """
    if isinstance(prod, str):
        name = prod.strip()
        link = ""
    elif isinstance(prod, dict):
        name = str(prod.get("name", "")).strip()
        link = str(prod.get("link", "")).strip()
    else:
        return None

    # Normalize special unicode spaces and superscripts
    name = (
        name.replace("\u202f", " ")
        .replace(" ", " ")
        .replace("⁻¹", "/L")
        .replace("L⁻¹", "L")
        .replace("–", "-")
        .strip()
    )

    if not name:
        return None

    # If link is broken, missing, or example.com, generate an active Google search link
    if not link or "example.com" in link or not (link.startswith("http://") or link.startswith("https://")):
        query = f"{name} fungicide agricultural treatment"
        link = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"

    return {"name": name, "link": link}

# ── Helper: Resolve Advisory ───────────────────────────────────────────────────
def get_disease_advisory(condition_key):
    if not condition_key:
        return {}

    cond_clean = condition_key.lower().strip().replace(" ", "_").replace("-", "_")

    raw_advisory = None
    # 1. Direct or Case-Insensitive Exact Match
    if condition_key in ADVISORY_DATA:
        raw_advisory = ADVISORY_DATA[condition_key]
    else:
        for k, v in ADVISORY_DATA.items():
            k_clean = k.lower().strip().replace(" ", "_").replace("-", "_")
            if k_clean == cond_clean:
                raw_advisory = v
                break

    # 2. Substring Match across Class Name or Disease
    if not raw_advisory:
        for k, v in ADVISORY_DATA.items():
            k_clean = k.lower().strip().replace(" ", "_").replace("-", "_")
            if k_clean in cond_clean or cond_clean in k_clean:
                raw_advisory = v
                break
            # Also check disease_name inside advisory
            d_name = v.get("disease_name", "").lower().strip().replace(" ", "_")
            if d_name and (d_name in cond_clean or cond_clean in d_name):
                raw_advisory = v
                break

    if raw_advisory:
        adv = dict(raw_advisory)
        if not adv.get("disease_name"):
            adv["disease_name"] = condition_key.replace("___", " - ").replace("_", " ").title()
    else:
        adv = {
            "disease_name": condition_key.replace("_", " ").title(),
            "causes": "Foliar lesion or necrotic pathogen observed on specimen surface.",
            "prevention": "Inspect adjacent crop canopy. Maintain optimal plant spacing to allow ventilation and foliage drying.",
            "treatment": "Consult regional agricultural extension office for appropriate therapeutic spraying.",
            "products": []
        }

    # Map symptoms to causes and build treatment description if needed
    if not adv.get("causes") and adv.get("symptoms"):
        adv["causes"] = adv["symptoms"]
    if not adv.get("treatment") and (adv.get("chemical") or adv.get("organic")):
        treat_parts = []
        if adv.get("chemical"):
            treat_parts.append(f"Chemical Control: {', '.join(adv['chemical'])}")
        if adv.get("organic"):
            treat_parts.append(f"Organic/Biological: {', '.join(adv['organic'])}")
        adv["treatment"] = ". ".join(treat_parts) + "."

    # Sanitize and validate all product recommendations
    sanitized_products = []
    for p in adv.get("products", []):
        cleaned = clean_product_entry(p)
        if cleaned:
            sanitized_products.append(cleaned)

    # If products list was not defined, use the scientific chemical and organic formulations
    if not sanitized_products:
        for c in adv.get("chemical", []):
            cleaned = clean_product_entry(c)
            if cleaned:
                sanitized_products.append(cleaned)
        for o in adv.get("organic", []):
            cleaned = clean_product_entry(o)
            if cleaned:
                sanitized_products.append(cleaned)

    # Dynamic fallback products if active disease but product array was empty
    if not sanitized_products:
        cond_check = cond_clean.lower()
        if any(d in cond_check for d in ["blight", "spot", "rot", "rust", "mold", "mildew", "virus", "mite"]):
            if "bacteri" in cond_check or "spot" in cond_check:
                sanitized_products = [
                    clean_product_entry({"name": "Copper Hydroxide 20 g/L (Kocide 3000)", "link": ""}),
                    clean_product_entry({"name": "Chlorothalonil 500 g/L Protectant", "link": ""})
                ]
            elif "rust" in cond_check:
                sanitized_products = [
                    clean_product_entry({"name": "Propiconazole 80 g/L (Tilt Fungicide)", "link": ""}),
                    clean_product_entry({"name": "Pyraclostrobin 20 g/L (Headline)", "link": ""})
                ]
            elif "mildew" in cond_check:
                sanitized_products = [
                    clean_product_entry({"name": "Wettable Sulfur 80% WDG", "link": ""}),
                    clean_product_entry({"name": "Myclobutanil 40 g/L (Rally 40W)", "link": ""})
                ]
            elif "virus" in cond_check or "mite" in cond_check:
                sanitized_products = [
                    clean_product_entry({"name": "Cold-Pressed Neem Oil Miticide", "link": ""}),
                    clean_product_entry({"name": "Insecticidal Potassium Soap 5%", "link": ""})
                ]
            else:
                sanitized_products = [
                    clean_product_entry({"name": "Mancozeb 75% WP (Dithane M-45)", "link": ""}),
                    clean_product_entry({"name": "Copper Oxychloride 50 g/L", "link": ""})
                ]

    adv["products"] = sanitized_products
    return adv

# ── Routes ─────────────────────────────────────────────────────────────────────

# 1. Commercial Landing Page
@app.route("/")
def index():
    return render_template("index.html")

# 2. Farmer Authentication: Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("diagnostics"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter both username and password.", "warning")
            return render_template("login.html")

        if login_user:
            user_dict, msg = login_user(username, password)
            if user_dict:
                session["user"] = user_dict
                flash(f"Welcome back, {user_dict['full_name']}! Signed into your farm portal.", "success")
                return redirect(url_for("diagnostics"))
            else:
                flash(msg, "danger")
        else:
            flash("Authentication database service unavailable.", "danger")

    return render_template("login.html")

# 3. Farmer Authentication: Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user"):
        return redirect(url_for("diagnostics"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        farm_name = request.form.get("farm_name", "").strip()
        location = request.form.get("location", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not full_name or not username or not password:
            flash("Full name, username, and password are required.", "warning")
            return render_template("register.html", full_name=full_name, farm_name=farm_name, location=location, username=username)

        # 1. Full name validation: numbers or signs cannot be entered
        if not re.match(r"^[A-Za-z\s]+$", full_name) or len(full_name) < 2:
            flash("Full Name cannot contain numbers or special characters.", "warning")
            return render_template("register.html", full_name=full_name, farm_name=farm_name, location=location, username=username)

        # 2. District / Location validation: valid location, not any random numbers
        if location:
            if re.search(r"\d", location) or not re.match(r"^[A-Za-z\s,.-]+$", location) or len(location) < 2:
                flash("District / Location must be a valid location name and cannot contain numbers.", "warning")
                return render_template("register.html", full_name=full_name, farm_name=farm_name, location=location, username=username)

        # 3. Choose Username validation: must start with an alphabetical character
        if not re.match(r"^[A-Za-z]", username):
            flash("Username must start with an alphabetical character.", "warning")
            return render_template("register.html", full_name=full_name, farm_name=farm_name, location=location, username=username)

        if register_user:
            ok, msg = register_user(username, password, full_name, farm_name, location)
            if ok:
                flash("🌱 Registration successful! You can now log into your account.", "success")
                return redirect(url_for("login"))
            else:
                flash(msg, "danger")
        else:
            flash("Registration service unavailable.", "danger")

        return render_template("register.html", full_name=full_name, farm_name=farm_name, location=location, username=username)

    return render_template("register.html")

# 4. Farmer Authentication: Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been signed out of your farm portal session.", "success")
    return redirect(url_for("index"))

# 5. Farm Scan Audit History (Protected)
@app.route("/history")
def farm_history():
    if not session.get("user"):
        flash("🔒 You must be logged in to view your private farm audit logs.", "warning")
        return redirect(url_for("login"))

    user = session["user"]
    scans = []
    if get_user_scans:
        try:
            scans = get_user_scans(user["id"])
        except Exception as e:
            flash(f"Could not load scan logs: {e}", "danger")

    total_scans = len(scans)
    critical_count = len([s for s in scans if s.get("severity") == "Critical"])
    healthy_count = len([s for s in scans if s.get("severity") == "Healthy"])

    return render_template(
        "history.html",
        user=user,
        scans=scans,
        total_scans=total_scans,
        critical_count=critical_count,
        healthy_count=healthy_count
    )

# 5b. Serve Saved Crop Scan Images
@app.route("/user_scans/<path:filename>")
def serve_user_scan(filename):
    if not session.get("user"):
        flash("🔒 Authentication required to view archived scan images.", "warning")
        return redirect(url_for("login"))
    scans_dir = PROJECT_ROOT / "user_scans"
    return send_from_directory(scans_dir, filename)

# 6. Field Diagnostics (Guest & Logged-In Mode)
@app.route("/diagnostics", methods=["GET", "POST"])
def diagnostics():
    if request.method == "POST":
        uploaded = request.files.get("leaf_image")
        if not uploaded or uploaded.filename == "":
            flash("Please upload a leaf photograph before running diagnostics.", "warning")
            return redirect(url_for("diagnostics"))

        # Extract Form Controls
        manual_crop = request.form.get("manual_crop", "Auto-Detect via Field Vision")
        plot_id = request.form.get("plot_id", "").strip()
        # Standard calibrated operational threshold
        conf_threshold = 0.15

        # Always run the full Two-Stage cascaded pipeline by default
        use_two_stage = True if TWO_STAGE_AVAILABLE else False
        crop_override = None if manual_crop == "Auto-Detect via Field Vision" else manual_crop

        # Read Image into PIL
        try:
            image = Image.open(uploaded.stream).convert("RGB")
            # Fast downsample for instantaneous CPU inference
            max_dim = 800
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
        except Exception as e:
            flash(f"Invalid image file: {e}", "danger")
            return redirect(url_for("diagnostics"))

        # Run AI Inference Pipeline
        if predict_image is None:
            flash("Inference model weights not loaded. Please verify ResNet-50 weights exist.", "danger")
            return redirect(url_for("diagnostics"))

        try:
            if TWO_STAGE_AVAILABLE:
                result = predict_image(
                    image,
                    conf_threshold=conf_threshold,
                    use_two_stage=use_two_stage,
                    manual_crop=crop_override
                )
            else:
                result = predict_image(image, conf_threshold=conf_threshold)
        except Exception as exc:
            flash(f"Inference error during model execution: {exc}", "danger")
            return redirect(url_for("diagnostics"))

        # Compute Health Score & Metrics
        top_conf = result.get("highest_confidence", 0.0)
        detections = result.get("detections", [])
        num_regions = len(detections)

        health_score, health_tier, health_color = compute_canopy_health_score(
            result.get("primary_condition", ""), detections, top_conf
        )

        is_healthy = health_tier == "OPTIMAL"

        if is_healthy:
            severity_label = "🟢 Healthy / No Pathogen Detected"
            sev_class, sev_clean = "severity-low", "Healthy"
            diagnosis_clean = "Healthy Foliage"
        elif health_tier == "CRITICAL":
            severity_label = "🔴 Critical — Immediate Intervention Required"
            sev_class, sev_clean = "severity-critical", "Critical"
            diagnosis_clean = result.get("primary_condition", "Unknown Condition").replace("_", " ").title()
        elif health_tier == "VULNERABLE":
            severity_label = "🟡 Moderate — Prompt Attention Advised"
            sev_class, sev_clean = "severity-moderate", "Moderate"
            diagnosis_clean = result.get("primary_condition", "Unknown Condition").replace("_", " ").title()
        else:
            severity_label = "🟢 Minor Infection — Regular Field Scouting"
            sev_class, sev_clean = "severity-low", "Low"
            diagnosis_clean = result.get("primary_condition", "Unknown Condition").replace("_", " ").title()

        hotspot_val = "0 (Clean Leaf)" if is_healthy else f"{num_regions} Region(s)"
        badge_bg = "#E8F5E9" if health_tier == "OPTIMAL" else ("#FEF3C7" if health_tier == "VULNERABLE" else "#FEE2E2")

        # Prepare Base64 Images for Side-by-Side Inspection
        original_b64 = image_to_base64(image)
        annotated_b64 = image_to_base64(result.get("annotated_image", image))

        # Auto-save scan if farmer is logged in
        saved_record_id = None
        if session.get("user") and save_scan:
            try:
                # Store crop specimen image (annotated if generated, else uploaded raw PIL image)
                specimen_img = result.get("annotated_image") if result.get("annotated_image") is not None else image
                saved_record_id = save_scan(
                    user_id=session["user"]["id"],
                    plot_id=plot_id,
                    crop_species=result.get("primary_plant", "Unknown"),
                    condition=diagnosis_clean,
                    confidence=top_conf if not is_healthy else 0.95,
                    severity=sev_clean,
                    hotspots=0 if is_healthy else num_regions,
                    image_input=specimen_img
                )
            except Exception as e:
                print(f"[Warning] Failed to persist scan: {e}")

        # Retrieve Advisory Dossier
        advisory_info = {}
        if not is_healthy and detections:
            top_detection = max(detections, key=lambda d: d.get("confidence", 0))
            advisory_info = get_disease_advisory(top_detection.get("class", ""))
        elif not is_healthy:
            advisory_info = get_disease_advisory(result.get("primary_condition", ""))

        # Format bullet list for prevention if string with dashes
        if advisory_info and "prevention" in advisory_info and isinstance(advisory_info["prevention"], str):
            prev_lines = [
                line.strip().lstrip("-").strip()
                for line in advisory_info["prevention"].splitlines()
                if line.strip()
            ]
            if len(prev_lines) > 1 or any(l.startswith("-") for l in advisory_info["prevention"].splitlines()):
                advisory_info["prevention_list"] = prev_lines

        # Generate Text Field Report
        now_ts = datetime.now()
        report_text = (
            f"AGRISCAN FIELD REPORT\n"
            f"=========================================\n"
            f"Date / Time     : {now_ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Plot Identifier : {plot_id or 'General Specimen / Unspecified'}\n"
            f"Host Species    : {result.get('primary_plant', 'Unknown')}\n"
            f"Diagnosis       : {diagnosis_clean}\n"
            f"Health Score    : {health_score}% ({health_tier})\n"
            f"Severity Index  : {sev_clean}\n"
            f"Foliar Status   : {'Healthy Foliage' if is_healthy else 'Pathogen Detected'}\n"
            f"Diagnostic Mode : Precision Foliar Scan\n"
            f"=========================================\n"
        )

        return render_template(
            "diagnostics.html",
            result=result,
            selected_crop=manual_crop,
            plot_id=plot_id,
            use_two_stage=use_two_stage,
            health_score=health_score,
            health_tier=health_tier,
            health_color=health_color,
            badge_bg=badge_bg,
            is_healthy=is_healthy,
            severity_label=severity_label,
            sev_class=sev_class,
            sev_clean=sev_clean,
            diagnosis_clean=diagnosis_clean,
            hotspot_val=hotspot_val,
            original_image_data=original_b64,
            annotated_image_data=annotated_b64,
            detections=detections,
            advisory=advisory_info,
            report_text=report_text,
            now_str=now_ts.strftime("%Y%m%d_%H%M%S"),
            saved_record_id=saved_record_id
        )

    # GET Request: Initial View
    return render_template(
        "diagnostics.html",
        result=None,
        selected_crop="Auto-Detect via Field Vision",
        plot_id="",
        use_two_stage=TWO_STAGE_AVAILABLE
    )

# ── Application Entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
