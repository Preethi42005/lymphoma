"""Auth API — stores signup/login user details in users.xlsx"""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).resolve().parent
USERS_FILE = ROOT / "users.xlsx"
COLUMNS = [
    "id",
    "name",
    "phone",
    "medicalHistory",
    "email",
    "password",
    "createdAt",
    "lastLoginAt",
]


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _load_users() -> pd.DataFrame:
    if USERS_FILE.exists():
        df = pd.read_excel(USERS_FILE).fillna("")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _load_login_history() -> pd.DataFrame:
    history_cols = ["email", "name", "action", "success", "timestamp"]
    if not USERS_FILE.exists():
        return pd.DataFrame(columns=history_cols)
    try:
        df = pd.read_excel(USERS_FILE, sheet_name="LoginHistory").fillna("")
        for col in history_cols:
            if col not in df.columns:
                df[col] = ""
        return df[history_cols]
    except Exception:
        return pd.DataFrame(columns=history_cols)


def _save_workbook(users: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> None:
    users_out = users.copy()
    for col in COLUMNS:
        users_out[col] = users_out[col].map(lambda v: "" if pd.isna(v) else str(v))
    history_out = history if history is not None else _load_login_history()
    with pd.ExcelWriter(USERS_FILE, engine="openpyxl") as writer:
        users_out.to_excel(writer, sheet_name="Users", index=False)
        history_out.to_excel(writer, sheet_name="LoginHistory", index=False)


def _save_users(df: pd.DataFrame) -> None:
    _save_workbook(df)


def _append_history(email: str, name: str, action: str, success: bool) -> None:
    history = _load_login_history()
    row = {
        "email": email,
        "name": name,
        "action": action,
        "success": "yes" if success else "no",
        "timestamp": datetime.utcnow().isoformat(),
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    users = _load_users()
    _save_workbook(users, history)


def _user_public(row: pd.Series) -> dict:
    return {
        "id": _cell(row["id"]),
        "name": _cell(row["name"]),
        "phone": _cell(row["phone"]),
        "medicalHistory": _cell(row["medicalHistory"]),
        "email": _cell(row["email"]),
        "createdAt": _cell(row["createdAt"]),
        "lastLoginAt": _cell(row.get("lastLoginAt", "")),
    }


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    medical_history = (data.get("medicalHistory") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not all([name, phone, medical_history, email, password]):
        return jsonify({"success": False, "error": "All fields are required"}), 400

    users = _load_users()
    if not users.empty and (users["email"].str.lower() == email).any():
        return jsonify({"success": False, "error": "Email already registered"}), 400

    user_id = f"user_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    now = datetime.utcnow().isoformat()
    new_row = {
        "id": user_id,
        "name": name,
        "phone": phone,
        "medicalHistory": medical_history,
        "email": email,
        "password": _hash_password(password),
        "createdAt": now,
        "lastLoginAt": now,
    }
    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
    _save_users(users)
    _append_history(email, name, "signup", True)

    return jsonify({
        "success": True,
        "user": {
            "id": user_id,
            "name": name,
            "phone": phone,
            "medicalHistory": medical_history,
            "email": email,
            "createdAt": now,
            "lastLoginAt": now,
        },
    })


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required"}), 400

    users = _load_users()
    if users.empty:
        return jsonify({"success": False, "error": "Invalid email or password"}), 401

    match = users[users["email"].str.lower() == email]
    if match.empty:
        _append_history(email, "", "login", False)
        return jsonify({"success": False, "error": "Invalid email or password"}), 401

    idx = match.index[0]
    row = users.loc[idx]
    if str(row["password"]) != _hash_password(password):
        _append_history(email, _cell(row["name"]), "login", False)
        return jsonify({"success": False, "error": "Invalid email or password"}), 401

    last_login = datetime.utcnow().isoformat()
    users.at[idx, "lastLoginAt"] = last_login
    _save_users(users)
    _append_history(email, _cell(row["name"]), "login", True)

    user = _user_public(users.loc[idx])
    user["lastLoginAt"] = last_login
    return jsonify({"success": True, "user": user})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "usersFile": str(USERS_FILE)})


if __name__ == "__main__":
    print(f"Auth API storing users in: {USERS_FILE}")
    app.run(host="0.0.0.0", port=5002, debug=True)
