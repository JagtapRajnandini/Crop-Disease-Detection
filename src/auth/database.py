"""
src/auth/database.py
--------------------
SQLite Database & Authentication Engine for AgriScan AI SaaS.
Handles user registration, password hashing (SHA-256), session verification,
and field scan history persistence.
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
try:
    import numpy as np
except ImportError:
    np = None

# Database file location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "agriscan_users.db"
SCANS_DIR = PROJECT_ROOT / "user_scans"

def get_connection():
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Hashes passwords using SHA-256 with a project salt."""
    salt = "agriscan_ai_cummins_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def init_db():
    """Initializes tables for users and farm scans."""
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                farm_name TEXT,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Scans Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plot_id TEXT,
                crop_species TEXT NOT NULL,
                primary_condition TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                hotspots_count INTEGER NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

# ── User Authentication Functions ──────────────────────────────────────────────

def register_user(username, password, full_name, farm_name="", location=""):
    """
    Registers a new farmer.
    Returns: (success: bool, message: str)
    """
    username = username.strip().lower()
    if not username or not password or not full_name:
        return False, "Username, password, and full name are required."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    pwd_hash = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, farm_name, location)
                VALUES (?, ?, ?, ?, ?)
            """, (username, pwd_hash, full_name.strip(), farm_name.strip(), location.strip()))
            conn.commit()
            return True, "Account registered successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose a different one."
    except Exception as e:
        return False, f"Registration failed: {e}"

def login_user(username, password):
    """
    Authenticates a user.
    Returns: (user_dict or None, message: str)
    """
    username = username.strip().lower()
    pwd_hash = hash_password(password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, full_name, farm_name, location, created_at
            FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, pwd_hash))
        row = cursor.fetchone()

        if row:
            return dict(row), "Login successful!"
        else:
            return None, "Invalid username or password."

# ── Scan History Functions ─────────────────────────────────────────────────────

def save_scan(user_id, plot_id, crop_species, condition, confidence, severity, hotspots, image_input=None, image_np=None):
    """
    Saves a diagnostic scan record and the crop image.
    Supports PIL Image or numpy array (RGB).
    Returns scan_id
    """
    saved_img_rel_path = None
    target_img = image_input if image_input is not None else image_np

    if target_img is not None:
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_filename = f"scan_u{user_id}_{timestamp_str}.jpg"
            abs_img_path = SCANS_DIR / img_filename

            # If PIL Image
            if hasattr(target_img, "save"):
                target_img.save(str(abs_img_path), format="JPEG", quality=90)
                saved_img_rel_path = f"user_scans/{img_filename}"
            # If numpy array
            elif np is not None and isinstance(target_img, np.ndarray):
                import cv2
                bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(abs_img_path), bgr)
                saved_img_rel_path = f"user_scans/{img_filename}"
        except Exception as e:
            print(f"[Warning] Failed to save crop specimen image to disk: {e}")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scans (
                user_id, plot_id, crop_species, primary_condition,
                confidence, severity, hotspots_count, image_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            plot_id if plot_id else "Unspecified Plot",
            crop_species,
            condition,
            round(float(confidence), 4),
            severity,
            int(hotspots),
            saved_img_rel_path
        ))
        conn.commit()
        return cursor.lastrowid

def get_user_scans(user_id):
    """
    Retrieves all past scans for a specific farmer, sorted newest first.
    Returns list of dicts.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, plot_id, crop_species, primary_condition,
                   confidence, severity, hotspots_count, image_path, created_at
            FROM scans
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        scans = []
        for r in rows:
            d = dict(r)
            if d.get("image_path"):
                clean_path = str(d["image_path"]).replace("\\", "/")
                # Ensure the file actually exists on disk before serving
                if (PROJECT_ROOT / clean_path).exists():
                    d["image_path"] = clean_path
                else:
                    d["image_path"] = None
            scans.append(d)
        return scans

# Automatically initialize tables when imported
init_db()