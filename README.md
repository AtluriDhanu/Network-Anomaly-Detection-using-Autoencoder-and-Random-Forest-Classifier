# Network Anomaly Detection using Autoencoder and Random Forest Classifier
A network anomaly detection system that learns normal traffic patterns and flags suspicious behaviour. It uses autoencoder and to understand regular network activity and random forst classifier to catch anomalies more accurately. The system processes traffic data with Scapy and Pandas, applies normalization and shows real-time results with clear visualizations in Matplotlib and Seaborn

## Features
- Real-time packet capture and anomaly detection.
- Autoencoder-based feature learning with Random Forest Classification.
- Synthetic traffic simulation for testing.
- Flask-based interactive dashboard for anomaly visualization.
- Performance metrics including accuracy, F1-score and Precision-recall.
- Real-time alerts for anomalous connections.

## Workflow Diagram
train_classifier.py (Train Autoencoder and Random Forest Classifier) ---> new_detector.py (Load trained model & analyze traffic dataset) ---> simulate_traffic (Generate synthetic traffic like Normal, SYN Flood, UDP Flood, ICMP Flood & Port Scan) ---> live_traffic.py (Capture live or simulated traffic & classify anomalies) ---> dashboard.py (Flask + Dash dashboard with real-time stats and graphs) 

## Steps to run
- Install the required requirements.  (pip install -r requirements.txt)
- Train the model, if you need to retrain the system. This step is resource-intensive and may take time. (python train_classifier.py) 
- Run the detector on dataset, it loads the trained model, processes traffic data and output metrics like accuracy and F1 Score. (python new_detector.py)
- Simulate the traffic, choose from different modes like Normal, SYN Flood, UDP Flood, ICMP Flood, Port Scan. Enter the source and target IP address, MAC address and duration. (python simulate_traffic)
- Start Live Detection, captures the simulated or real traffic, classifies the anomalies and exposes results via a Flask API (python live_traffic.py)
- Launch the dashboard, opens an interactive dashboard at "http://127.0.0.1:8051" with Anomaly score trends, Anomally frequency over time and Anomaly breakdown by type (python dashboard.py)
