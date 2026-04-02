import os
import random
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "secretkey"

# ==============================
# LOAD MODEL + PREPROCESSOR
# ==============================
model = joblib.load("rf_model.joblib")
preprocessor = joblib.load("preprocessing_engine.joblib")
feature_columns = joblib.load("feature_columns.pkl")

# ==============================
# GLOBAL STORAGE
# ==============================
alerts = []
processed_data = []

# ==============================
# HOME
# ==============================
@app.route("/")
def home():
    return render_template("home.html")


# ==============================
# UPLOAD DATASET
# ==============================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global alerts, processed_data

    if request.method == "POST":
        file = request.files["dataset"]

        if not file:
            flash("No file selected")
            return redirect(request.url)

        df = pd.read_csv(file)

        # Align columns with training
        df = pd.get_dummies(df)

        for col in feature_columns:
            if col not in df:
                df[col] = 0

        df = df[feature_columns]

        # Preprocess + Predict
        X = preprocessor.transform(df)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        processed_data = []
        alerts = []

        for i in range(len(df)):
            predicted_class = predictions[i]
            confidence = max(probabilities[i]) * 100

            threat_status = "Threat" if predicted_class != "normal" else "Normal"

            processed_data.append({
                "duration": df.iloc[i].get("duration", 0),
                "src_bytes": df.iloc[i].get("src_bytes", 0),
                "dst_bytes": df.iloc[i].get("dst_bytes", 0),
                "count": df.iloc[i].get("count", 0),
                "srv_count": df.iloc[i].get("srv_count", 0),
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "threat_status": threat_status
            })

            # 🔥 CLEAN ALERT (NO IP LOGIC)
            if threat_status == "Threat":
                severity = "high" if confidence > 85 else "medium"

                alerts.append({
                    "title": f"{predicted_class} detected",
                    "message": f"Packets: {df.iloc[i].get('count',0)} | Confidence: {round(confidence,2)}%",
                    "severity": severity,
                    "x": random.randint(10, 90),
                    "y": random.randint(10, 90),
                    "place": random.choice([
                        "North Gateway", "East Node", "South Cluster",
                        "West Access", "Core Network"
                    ])
                })

        flash("Analysis completed successfully")
        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# ==============================
# DASHBOARD
# ==============================
@app.route("/dashboard")
def dashboard():
    total_logs = len(processed_data)
    threats = sum(1 for d in processed_data if d["threat_status"] == "Threat")
    safe = total_logs - threats

    avg_conf = (
        sum(d["confidence"] for d in processed_data) / total_logs
        if total_logs > 0 else 0
    )

    return render_template(
        "dashboard.html",
        total_logs=total_logs,
        threats=threats,
        safe=safe,
        avg_conf=round(avg_conf, 2),
        data=processed_data,
        alerts=alerts
    )


# ==============================
# ALERTS PAGE
# ==============================
@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", alerts=alerts)


# ==============================
# LIVE MONITOR (SIMULATION)
# ==============================
@app.route("/live")
def live():
    sample_data = processed_data[:20] if processed_data else []

    live_packets = len(sample_data)
    threats = sum(1 for d in sample_data if d["threat_status"] == "Threat")
    safe = live_packets - threats

    return render_template(
        "live.html",
        rows=sample_data,
        live_packets=live_packets,
        threats=threats,
        safe=safe,
        top_attack=sample_data[0]["predicted_class"] if sample_data else "normal"
    )


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
