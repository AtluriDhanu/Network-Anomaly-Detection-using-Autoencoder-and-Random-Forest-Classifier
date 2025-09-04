import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler  
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier  

# File Paths
TRAINING_SET_PATH = "C:/Users/arsha/Desktop/final sem project/network anomaly detection/Data/training-set.csv"
TESTING_SET_PATH = "C:/Users/arsha/Desktop/final sem project/network anomaly detection/Data/testing-set.csv"
MODEL_PATH = "Models/autoencoder_for_features.keras"
BOTTLENECK_MODEL_PATH = "Models/bottleneck_model.keras"
SCALER_PATH = "Models/data_scaler.pkl"
ENCODER_COLUMNS_PATH = "Models/encoder_columns.pkl"
CLASSIFIER_MODEL_PATH = "Models/supervised_classifier.pkl"
DETECTION_RESULTS_PATH = "Data/new_data_anomalies.csv"

# Ensure necessary folders exist
os.makedirs("Data", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# Preprocess Data
def preprocess_data(data_path, scaler_path, encoder_columns_path):
    print(f"Loading and preprocessing data from {data_path}...")
    try:
        df = pd.read_csv(data_path, header=0)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

    scaler = None
    encoder_columns = None
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_columns_path, "rb") as f:
            encoder_columns = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading necessary files for preprocessing: {e}")
        print(f"Please ensure {scaler_path} and {encoder_columns_path} exist.")
        return None

    if scaler is None or encoder_columns is None:
        print("Could not load scaler or encoder columns. Please check the files.")
        return None

    df = df.drop(columns=["id", "attack_cat"], errors="ignore")

    # Ensure all columns in encoder_columns are present in the test data
    missing_cols = [col for col in encoder_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in test data: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            df[col] = 0

    # Reorder columns to match the order used during training
    try:
        df = df[encoder_columns]
        print("Columns reordered to match training data.")
    except KeyError as e:
        print(f" Error: Could not reorder columns. Missing column in test data: {e}")
        return None

    df.fillna("Unknown", inplace=True)

    numeric_features = [
        "sbytes", "dbytes", "dpkts", "spkts", "rate", "sttl", "dttl", "smean", "dmean",
        "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd",
        "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "synack", "ackdat", "sjit",
        "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "sload", "dload", "sloss",
        "dloss", "sinpkt", "dinpkt"
    ]
    categorical_features = ["proto", "service", "state"]
    binary_features = ["is_sm_ips_ports"]

    # --- Feature Engineering ---
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-9)
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1e-9)
    if 'sbytes' in df.columns and 'spkts' in df.columns:
        df['bytes_per_pkt_src'] = df['sbytes'] / (df['spkts'] + 1e-9)
    if 'dbytes' in df.columns and 'dpkts' in df.columns:
        df['bytes_per_pkt_dst'] = df['dbytes'] / (df['dpkts'] + 1e-9)
    if 'ct_src_ltm' in df.columns and 'ct_dst_ltm' in df.columns:
        df['ct_ratio'] = df['ct_src_ltm'] / (df['ct_dst_ltm'] + 1e-9)

    new_numeric_features = ['bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_src', 'bytes_per_pkt_dst', 'ct_ratio']
    numeric_features = numeric_features + new_numeric_features

    # --- Categorical and Binary Feature Handling ---
    for col in categorical_features:
        if col in df.columns:
            try:
                if col in encoder_columns:
                    unique_values = df[col].unique()
                    mapping = {value: i for i, value in enumerate(unique_values)}
                    df[col] = df[col].map(mapping).fillna(-1)
                else:
                    print(f"Warning: Categorical column '{col}' not found in training encoder columns. Skipping encoding.")
            except Exception as e:
                print(f"Error loading encoder columns or encoding '{col}': {e}")
                print(f"Sample values in '{col}':\n{df[col].head()}")
                raise

    for col in binary_features:
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: 1 if x not in [0, "Unknown"] else 0)
            except Exception as e:
                print(f"Error processing binary column '{col}' in test data: {e}")
                print(f"Sample values in '{col}':\n{df[col].head()}")
                raise

    if numeric_features:
        try:
            df_numeric = df[numeric_features].fillna(0)
            if not df_numeric.empty:
                df[numeric_features] = scaler.transform(df_numeric)
                print(f"Numeric features in data scaled.")
            else:
                print("Warning: No numeric features to scale.")
        except Exception as e:
            print(f"Error during scaling of data: {e}")
            print("Ensure the numeric features match those used during training.")
            raise

    print(f"Preprocessed data shape (after alignment): {df.shape}")
    return df

# Perform Anomaly Detection
def detect_anomalies(new_data_path, model_path_encoder, classifier_save_path, results_path):
    print("Performing anomaly detection on new data...")
    preprocessed_data = preprocess_data(new_data_path, SCALER_PATH, ENCODER_COLUMNS_PATH)
    if preprocessed_data is None or preprocessed_data.empty:
        print("Could not preprocess new data or no data to analyze.")
        return None

    try:
        # Load the trained encoder model
        encoder = tf.keras.models.load_model(model_path_encoder)
        print(f"Loaded encoder model from {model_path_encoder}")

        # Load the trained classifier
        with open(classifier_save_path, 'rb') as f:
            classifier = pickle.load(f)
        print(f"Loaded supervised classifier from {classifier_save_path}")

        # Extract bottleneck features from the new data
        X_new_np = preprocessed_data.drop(columns=['label'], errors='ignore').to_numpy()
        if X_new_np.shape[1] == 0:
            print("Error: No features to predict on.")
            return None
        new_features = encoder.predict(X_new_np)
        print(f"Extracted bottleneck features for new data. Shape: {new_features.shape}")

        # Make predictions using the classifier
        predictions = classifier.predict(new_features)
        print(f"Performed anomaly prediction on new data.")

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({'prediction': predictions})
        results_df['prediction'] = results_df['prediction'].replace({0: 'Normal', 1: 'Anomaly'})

        # Save the predictions with the original data
        try:
            original_new_data = pd.read_csv(new_data_path)
            detection_results = pd.concat([original_new_data, results_df], axis=1)
            detection_results.to_csv(results_path, index=False)
            print(f"Anomaly detection results saved to {results_path}")
            return detection_results
        except Exception as e:
            print(f"Could not load original new data to merge with predictions: {e}. Predictions saved only.")
            results_df.to_csv(results_path, index=False)
            return results_df

    except FileNotFoundError as e:
        print(f"Error: Could not load necessary model or files: {e}")
        return None
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return None

# Evaluate Model
def evaluate_model(results_df_or_path):
    print("Evaluating the performance of the anomaly detection model...")
    results_df = None
    if isinstance(results_df_or_path, str):
        try:
            results_df = pd.read_csv(results_df_or_path)
        except FileNotFoundError:
            print(f"Error: Results file not found at {results_df_or_path}")
            return
    elif isinstance(results_df_or_path, pd.DataFrame):
        results_df = results_df_or_path
    else:
        print("Invalid input for evaluation. Expected a file path or a DataFrame.")
        return

    if results_df is None or 'label' not in results_df.columns or 'prediction' not in results_df.columns:
        print("Error: 'label' or 'prediction' column not found in the results.")
        return

    y_true = results_df['label']
    y_pred = results_df['prediction'].apply(lambda x: 1 if x == 'Anomaly' else 0)

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\n(TN, FP)")
    print("(FN, TP)")

    # Calculate Classification Report
    cr = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])
    print("\nClassification Report:")
    print(cr)

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

# Main Execution for Anomaly Detection and Evaluation
if __name__ == "__main__":
    detection_results = detect_anomalies(TESTING_SET_PATH, BOTTLENECK_MODEL_PATH, CLASSIFIER_MODEL_PATH, DETECTION_RESULTS_PATH)
    print("Anomaly detection process completed!")

    # Evaluate the model after detection
    if detection_results is not None and 'label' in detection_results.columns and 'prediction' in detection_results.columns:
        evaluate_model(detection_results)
        print("Evaluation process completed!")
    else:
        print("Anomaly detection did not complete successfully or results are missing 'label' and 'prediction' columns. Evaluation skipped.")


