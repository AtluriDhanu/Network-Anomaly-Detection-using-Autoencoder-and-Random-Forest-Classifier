import threading
import queue
import time
import random
from scapy.all import sniff, IP, TCP, UDP, Ether, sendp, wrpcap
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from flask import Flask, jsonify, redirect
import plotly.graph_objs as go  

# --- Global Variables ---
running = True
feature_queue = queue.Queue()
anomalies = []  

# --- Flask App for Dashboard ---
app = Flask(__name__)

@app.route('/')
def index():
    """Default route to confirm the API is running."""
    return "The Live Traffic Flask API is running. Visit /dashboard for the anomaly detection system."

@app.route('/dashboard')
def dashboard_redirect():
    """Redirect users to the running Dash dashboard."""
    return redirect("http://127.0.0.1:8051")

@app.route('/anomalies', methods=['GET'])
def get_anomalies():
    """API endpoint to expose detected anomalies for the dashboard."""
    try:
        return jsonify({
            "anomaly_scores": [
                {"timestamp": a["timestamp"], "score": a["score"], "category": a.get("category", "Unknown")}
                for a in anomalies
            ]
        })
    except Exception as e:
        print(f"Error serving anomalies: {e}")
        return jsonify({"error": "Failed to retrieve anomalies"}), 500  
# --- Graph Generation Function ---
def generate_anomaly_graph():
    """Generates an anomaly trend graph and saves it as an HTML file."""
    if not anomalies:
        print("No anomalies detected to plot.")
        return

    df = pd.DataFrame(anomalies)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["score"],
        mode="lines+markers", marker=dict(color="red"), name="Anomaly Score"
    ))
    fig.update_layout(title="Anomaly Score Over Time", xaxis_title="Timestamp", yaxis_title="Score (0-1)")

    fig.write_html("anomaly_graph.html")
    print("Anomaly graph saved as 'anomaly_graph.html'.")

# --- Simulated Anomaly Detection ---
def detect_anomalies():
    """Simulates anomaly detection by generating random data."""
    global anomalies
    while running:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        score = round(random.uniform(0, 1), 4)
        category = random.choice(["Category A", "Category B", "Category C"])

        if score > 0.6:  # Threshold for anomalies
            anomaly = {"timestamp": timestamp, "score": score, "category": category}
            anomalies.append(anomaly)
            print(f"Anomaly detected: {anomaly}")

            # Update graph each time an anomaly is detected
            generate_anomaly_graph()

            if len(anomalies) > 50:
                anomalies.pop(0)

        time.sleep(2)

# --- Start Anomaly Detection Thread ---
def start_anomaly_detection():
    """Starts the background thread for anomaly detection."""
    detection_thread = threading.Thread(target=detect_anomalies, daemon=True)
    detection_thread.start()

if __name__ == "__main__":
    start_anomaly_detection()
    app.run(debug=True, port=5000)
