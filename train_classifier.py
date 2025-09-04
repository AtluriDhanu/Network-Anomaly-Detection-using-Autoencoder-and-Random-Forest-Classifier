import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import train_test_split  

# File Paths
TRAINING_SET_PATH = "C:/Users/arsha/Desktop/final sem project/network anomaly detection/Data/training-set.csv"
MODEL_PATH = "Models/autoencoder_for_features.keras"
BOTTLENECK_MODEL_PATH = "Models/bottleneck_model.keras"
SCALER_PATH = "Models/data_scaler.pkl"
ENCODER_COLUMNS_PATH = "Models/encoder_columns.pkl"
CLASSIFIER_MODEL_PATH = "Models/supervised_classifier.pkl"

# Ensure necessary folders exist
os.makedirs("Models", exist_ok=True)

# Preprocess Training Data
def preprocess_training_data(training_path):
    print("Starting preprocess_training_data...")  
    print("Loading and preprocessing training dataset...")
    df = pd.read_csv(training_path)
    if df.empty:
        print(f"Error: Training dataset at {training_path} is empty.")
        return None

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

    df = df.drop(columns=["id", "attack_cat"], errors="ignore")
    df.fillna("Unknown", inplace=True)

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
                unique_values = df[col].unique()
                mapping = {value: i for i, value in enumerate(unique_values)}
                df[col] = df[col].map(mapping).fillna(-1)
                if df[col].dtype == 'object':
                    print(f"Warning: Still found object dtype in column '{col}'. Sample values: {df[col].unique()[:5]}")
                    df[col] = df[col].astype(str).str.lower().replace('[^a-z0-9]+', '', regex=True)
                    unique_values = df[col].unique()
                    mapping = {value: i for i, value in enumerate(unique_values)}
                    df[col] = df[col].map(mapping).fillna(-1)
                    if df[col].dtype == 'object':
                        print(f"Warning: Still object dtype after further cleaning in '{col}'.")
            except Exception as e:
                print(f"Error processing categorical column '{col}': {e}")
                print(f"Sample values in '{col}':\n{df[col].head()}")
                raise

    for col in binary_features:
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: 1 if x not in [0, "Unknown"] else 0)
            except Exception as e:
                print(f"Error processing binary column '{col}' in training data: {e}")
                print(f"Sample values in '{col}':\n{df[col].head()}")
                raise

    scaler = StandardScaler()
    df_numeric = df[numeric_features].fillna(0)
    if not df_numeric.empty:
        df[numeric_features] = scaler.fit_transform(df_numeric)
    else:
        print("Warning: No numeric features found after preprocessing.")

    encoder_columns = list(df.drop(columns=["label"]).columns)
    with open(ENCODER_COLUMNS_PATH, "wb") as f:
        pickle.dump(encoder_columns, f)
    print(f"Encoder columns saved: {ENCODER_COLUMNS_PATH}")

    return df, scaler

# Train Autoencoder for Feature Extraction
def train_autoencoder_for_features(df):
    print("Training autoencoder for feature extraction...")
    if df is None or df.empty:
        print("Error: Training data is None or empty. Cannot train autoencoder.")
        return None, None

    training_data = df.drop(columns=["label"], errors="ignore").to_numpy()
    if training_data.shape[1] == 0:
        print("Error: No features to train autoencoder on.")
        return None, None

    input_dim = training_data.shape[1]
    encoding_dim = 16

    # Define the input layer
    input_layer = keras.layers.Input(shape=(input_dim,))

    # Define the encoder layers
    encoded = keras.layers.Dense(128, activation="relu")(input_layer)
    encoded = keras.layers.Dense(64, activation="relu")(encoded)
    bottleneck = keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck")(encoded)

    # Define the decoder layers
    decoded = keras.layers.Dense(64, activation="relu")(bottleneck)
    decoded = keras.layers.Dense(input_dim, activation="linear")(decoded)

    # Create the full autoencoder model
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    history = autoencoder.fit(training_data, training_data, epochs=50, batch_size=128, shuffle=True, verbose=1)
    autoencoder.save(MODEL_PATH)
    print(f"Autoencoder saved for feature extraction: {MODEL_PATH}")

    # Explicitly define the encoder model
    try:
        encoder = keras.Model(inputs=input_layer, outputs=bottleneck)
        encoder.save(BOTTLENECK_MODEL_PATH)
        print(f"Encoder (bottleneck) model saved: {BOTTLENECK_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving encoder model: {e}")
        print(f"Details: {e}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Models/autoencoder_loss.png')
    print("Autoencoder training loss plot saved to Models/autoencoder_loss.png")

    return encoder, df["label"].to_numpy()

# Train Supervised Classifier with Hyperparameter Tuning
def train_supervised_classifier(train_df, model_path_encoder, classifier_save_path):
    print("Training supervised classifier with hyperparameter tuning...")
    if train_df is None or train_df.empty:
        print("Error: Training data for classifier is None or empty.")
        return

    X_train = train_df.drop(columns=['label'], errors='ignore')
    y_train = train_df.get('label')

    if X_train.empty or y_train is None or y_train.empty:
        print("Error: No training data for the classifier.")
        return

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    try:
        try:
            encoder = tf.keras.models.load_model(model_path_encoder)
            print(f"Loaded encoder model from {model_path_encoder}")
            # Extract bottleneck features using the loaded encoder
            train_features = encoder.predict(X_train_np)

            # Define the parameter grid for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4]
            }

            # Initialize the Random Forest Classifier
            rf_classifier = RandomForestClassifier(random_state=42)

            # Perform GridSearchCV
            grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                                       cv=3, scoring='f1', n_jobs=-1, verbose=2)
            grid_search.fit(train_features, y_train_np)

            # Get the best parameters and the best estimator
            best_params = grid_search.best_params_
            best_rf_classifier = grid_search.best_estimator_
            print(f"Best hyperparameters found: {best_params}")

            # Save the trained classifier with the best hyperparameters
            with open(classifier_save_path, 'wb') as f:
                pickle.dump(best_rf_classifier, f)
            print(f"Best supervised classifier (with tuned hyperparameters) saved to {classifier_save_path}")

        except FileNotFoundError:
            print(f"Error: Encoder model not found at {model_path_encoder}. Please ensure it's trained and saved.")
        except Exception as e:
            print(f"Error during encoder processing or classifier training: {e}")

    except Exception as e:
        print(f"Error during classifier training (outer block): {e}")

# Main Execution for Training
if __name__ == "__main__":
    try:
        train_df, data_scaler = preprocess_training_data(TRAINING_SET_PATH)
        if train_df is not None:
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(data_scaler, f)
            encoder, _ = train_autoencoder_for_features(train_df)
            if encoder is not None:
                train_supervised_classifier(train_df, BOTTLENECK_MODEL_PATH, CLASSIFIER_MODEL_PATH)
                print("Training process with hyperparameter tuning completed successfully!")
            else:
                print("Autoencoder training failed, skipping classifier training.")
        else:
            print("Preprocessing of training data failed, skipping training.")
    except Exception as e:
        print(f"An error occurred during the training process: {e}")
