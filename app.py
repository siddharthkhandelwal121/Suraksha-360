# app.py
import os
import json
import time
import pickle
import hashlib
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

BASE_DIR = Path(__file__).parent
PERSIST_FILE = BASE_DIR / "data.json"
PUBLIC_DIR = BASE_DIR / "public"

# Use threading async mode (stable for development on Windows / recent Python)
socketio = None

# In-memory store (persisted to data.json if present)
store = {
    "users": [{"email": "admin@example.com", "password": "12345"}],
    "alerts": [],
    "notifications": [],
    "earthquake": {"magnitude": None, "time": None}
}

def load_persist():
    if PERSIST_FILE.exists():
        try:
            j = json.loads(PERSIST_FILE.read_text(encoding="utf8"))
            for k, v in j.items():
                store[k] = v
            print("Loaded persisted data.")
        except Exception as e:
            print("Failed to load persistence:", e)

def save_persist():
    try:
        PERSIST_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf8")
    except Exception as e:
        print("Failed to save persistence:", e)

load_persist()

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")
CORS(app)
# create socketio with threading mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

def mkid():
    return os.urandom(8).hex()

# Optional model files mapping (if you later add real models)
MODEL_FILES = {
    "flood": BASE_DIR / "flood_model.pkl",
    "landslide": BASE_DIR / "landslide_model.pkl",
    "earthquake": BASE_DIR / "earthquake_model.pkl"
}

# -------------------- deterministic forecast helper --------------------
def deterministic_seed_from(text):
    h = hashlib.sha256((text or "default").encode("utf8")).digest()
    return h[0]

def make_mock_forecast(location, disaster):
    """
    Returns deterministic 7-day forecast:
      { "location": str, "daily": [{day_index, day_label, probability}, ...] }
    Different scale/bias per disaster so patterns differ.
    """
    if not location:
        location = "Unknown"

    today = datetime.date.today()
    day_labels = [(today + datetime.timedelta(days=i)).strftime("%a") for i in range(7)]

    seed = deterministic_seed_from(f"{location}|{disaster}")
    base = (seed % 70) / 100.0

    if disaster == "earthquake":
        scale, bias = 0.35, 0.02
    elif disaster == "landslide":
        scale, bias = 0.9, 0.05
    else:  # flood
        scale, bias = 0.95, 0.08

    daily = []
    for i in range(7):
        jitter = ((seed * (i + 1)) % 37) / 300.0
        raw = (base + jitter) * scale + bias
        prob = round(max(0.01, min(0.99, raw)), 3)
        daily.append({"day_index": i, "day_label": day_labels[i], "probability": prob})

    return {"location": location, "daily": daily}

# -------------------- Serve frontend (public/ or root index.html) --------------------
@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve_frontend(path):
    # try public/
    if PUBLIC_DIR.exists():
        if path == "" or path == "index.html":
            target = PUBLIC_DIR / "index.html"
            if target.exists():
                resp = make_response(send_from_directory(str(PUBLIC_DIR), "index.html"))
                resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                return resp
        else:
            candidate = PUBLIC_DIR / path
            if candidate.exists():
                return send_from_directory(str(PUBLIC_DIR), path)

    # fallback root index.html
    if path == "" or path == "index.html":
        root_index = BASE_DIR / "index.html"
        if root_index.exists():
            resp = make_response(send_from_directory(str(BASE_DIR), "index.html"))
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return resp

    candidate_root = BASE_DIR / path
    if candidate_root.exists():
        return send_from_directory(str(BASE_DIR), path)

    # API or socket paths -> 404 (so API handlers run)
    if path.startswith("api/") or path.startswith("socket.io/"):
        return jsonify(ok=False, message="Not found"), 404

    # SPA fallback
    fallback = PUBLIC_DIR / "index.html"
    if fallback.exists():
        return send_from_directory(str(PUBLIC_DIR), "index.html")
    fallback2 = BASE_DIR / "index.html"
    if fallback2.exists():
        return send_from_directory(str(BASE_DIR), "index.html")
    return "Index file not found on server.", 404

# -------------------- REST API --------------------
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()
    if not email or not password:
        return jsonify(ok=False, message="Missing credentials"), 400
    user = next((u for u in store["users"] if u["email"] == email and u["password"] == password), None)
    if not user:
        return jsonify(ok=False, message="Invalid credentials"), 401
    return jsonify(ok=True, user={"email": email})

@app.route("/api/alerts", methods=["GET"])
def api_get_alerts():
    return jsonify(ok=True, alerts=store["alerts"])

@app.route("/api/alerts", methods=["POST"])
def api_add_alert():
    data = request.get_json(silent=True) or {}
    type_ = (data.get("type") or "").strip()
    location = (data.get("location") or "").strip()
    level = (data.get("level") or "").strip()
    if not type_ or not location or not level:
        return jsonify(ok=False, message="Missing fields"), 400
    a = {"id": mkid(), "type": type_, "location": location, "level": level}
    store["alerts"].append(a)
    save_persist()
    socketio.emit("alertAdded", a)
    return jsonify(ok=True, alert=a)

@app.route("/api/alerts/<aid>", methods=["DELETE"])
def api_delete_alert(aid):
    before = len(store["alerts"])
    store["alerts"] = [a for a in store["alerts"] if a["id"] != aid]
    if len(store["alerts"]) == before:
        return jsonify(ok=False, message="Not found"), 404
    save_persist()
    socketio.emit("alertRemoved", {"id": aid})
    return jsonify(ok=True)

@app.route("/api/alerts", methods=["DELETE"])
def api_delete_all_alerts():
    store["alerts"] = []
    save_persist()
    socketio.emit("alertsCleared")
    return jsonify(ok=True)

@app.route("/api/notifications", methods=["GET"])
def api_get_notifications():
    return jsonify(ok=True, notifications=store["notifications"])

@app.route("/api/notifications", methods=["POST"])
def api_add_notification():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(ok=False, message="Missing text"), 400
    n = {"id": mkid(), "text": text, "time": int(time.time() * 1000)}
    store["notifications"].append(n)
    save_persist()
    socketio.emit("notificationAdded", n)
    return jsonify(ok=True, notification=n)

@app.route("/api/notifications/<nid>", methods=["DELETE"])
def api_delete_notification(nid):
    store["notifications"] = [n for n in store["notifications"] if n["id"] != nid]
    save_persist()
    socketio.emit("notificationRemoved", {"id": nid})
    return jsonify(ok=True)

@app.route("/api/earthquake/simulate", methods=["POST"])
def api_simulate_quake():
    data = request.get_json(silent=True) or {}
    mag = data.get("magnitude")
    try:
        if mag is not None:
            mag = float(mag)
        else:
            mag = round((os.urandom(1)[0] / 255) * 4 + 3, 1)
    except Exception:
        mag = round((os.urandom(1)[0] / 255) * 4 + 3, 1)
    store["earthquake"] = {"magnitude": mag, "time": int(time.time() * 1000)}
    n = {"id": mkid(), "text": f"Earthquake {mag} magnitude detected", "time": int(time.time() * 1000)}
    store["notifications"].append(n)
    save_persist()
    socketio.emit("earthquake", store["earthquake"])
    socketio.emit("notificationAdded", n)
    return jsonify(ok=True, earthquake=store["earthquake"])

@app.route("/api/sos", methods=["POST"])
def api_sos():
    data = request.get_json(silent=True) or {}
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify(ok=False, message="Missing coords"), 400
    n = {"id": mkid(), "text": f"SOS: {lat}, {lon} — help requested", "time": int(time.time() * 1000)}
    store["notifications"].append(n)
    save_persist()
    socketio.emit("notificationAdded", n)
    return jsonify(ok=True, message="SOS received")

@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(ok=True, alerts=store["alerts"], notifications=store["notifications"], earthquake=store["earthquake"])

# -------------------- Prediction endpoint --------------------
@app.route("/api/predict/<disaster>", methods=["GET"])
def api_predict(disaster):
    """
    Returns:
      { ok: True,
        activeAlerts: [{ location, probability }, ...],
        forecast: { location, daily: [{ day_index, probability, day_label }] }
      }
    """
    disaster = (disaster or "").lower()
    if disaster not in ("flood", "landslide", "earthquake"):
        return jsonify(ok=False, message="Unknown disaster"), 400

    # Pre-defined candidate locations for each disaster (editable)
    locations_map = {
        "flood": ["Assam", "Kolkata", "Mumbai", "Lucknow", "Chennai"],
        "landslide": ["Shimla", "Darjeeling", "Munnar", "Gangtok", "Shillong"],
        "earthquake": ["Delhi", "Gujarat", "Bihar", "Nepal Border", "North East"]
    }
    candidates = locations_map.get(disaster, ["Unknown"])

    # Optional location query param to focus forecast on one place
    requested_location = (request.args.get("location") or "").strip() or None
    if requested_location and requested_location not in candidates:
        # if user requested a location not in our list, allow it but still include candidates
        # for forecast purposes use the requested location specifically
        forecast = make_mock_forecast(requested_location, disaster)
    else:
        # default forecast place = requested_location or top candidate
        forecast_place = requested_location or candidates[0]
        forecast = make_mock_forecast(forecast_place, disaster)

    # Build active alerts list using day 0 (today) probability for each candidate
    active_alerts = []
    for city in candidates:
        fc = make_mock_forecast(city, disaster)
        today_prob = fc["daily"][0]["probability"] if fc and fc.get("daily") else 0.0
        active_alerts.append({"location": city, "probability": today_prob})

    # sort by probability desc
    active_alerts.sort(key=lambda x: x["probability"], reverse=True)

    return jsonify(ok=True, activeAlerts=active_alerts, forecast=forecast)

# -------------------- Socket.IO handlers --------------------
@socketio.on("connect")
def on_connect():
    print("socket connected")
    emit("state", {"alerts": store["alerts"], "notifications": store["notifications"], "earthquake": store["earthquake"]})

@socketio.on("getState")
def on_get_state():
    emit("state", {"alerts": store["alerts"], "notifications": store["notifications"], "earthquake": store["earthquake"]})

@socketio.on("disconnect")
def on_disconnect():
    print("socket disconnected")

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Starting server on http://0.0.0.0:{port}")
    socketio.run(app, host="0.0.0.0", port=port)
