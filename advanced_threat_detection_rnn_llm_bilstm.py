import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Data Generation (Simulated Data Security Logs) ---
# We'll simulate sequences of events related to data access.
# Features could include: user ID (categorical), file size, access type (read/write/delete),
# time since last access, data volume transferred.
# Anomalies will represent suspicious activities like large data transfers by unusual users
# or rapid access to many files.

def generate_simulated_security_data(num_sequences=1500, max_sequence_length=30, num_users=50, anomaly_ratio=0.07):
    """
    Generates simulated sequential data for data security anomaly detection.
    Each sequence represents a series of data access events.
    Features: [user_id_encoded, file_size_log, access_type_encoded, time_since_last_access_log, data_volume_log]
    """
    X = []
    y = [] # 0 for normal, 1 for anomaly

    # Simulate a pool of common user IDs and file sizes
    common_users = np.arange(num_users // 2) # Half the users are "common"
    rare_users = np.arange(num_users // 2, num_users) # Other half are "rare" but exist

    # Access types: 0=read, 1=write, 2=delete, 3=copy, 4=transfer_out
    access_types = [0, 1, 2, 3, 4]

    for _ in range(num_sequences):
        sequence_length = np.random.randint(5, max_sequence_length) # Variable sequence length
        current_sequence = []
        is_anomaly_sequence = False

        for i in range(sequence_length):
            # Normal behavior:
            user_id = np.random.choice(common_users)
            file_size = np.random.lognormal(mean=7, sigma=1.0) # Typical file sizes
            access_type = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]) # Mostly read/write
            time_since_last_access = np.random.exponential(scale=100) # Typical time gaps
            data_volume = np.random.lognormal(mean=5, sigma=0.5) # Typical data volume

            # Introduce anomalies for some sequences
            if np.random.rand() < anomaly_ratio and not is_anomaly_sequence:
                is_anomaly_sequence = True # Mark this sequence as anomalous

                # Inject anomalous events within the sequence
                if np.random.rand() < 0.6: # Large data transfer by rare user
                    user_id = np.random.choice(rare_users) # Unusual user
                    file_size = np.random.lognormal(mean=12, sigma=1.5) # Very large file
                    access_type = 4 # Data transfer out
                    data_volume = np.random.lognormal(mean=10, sigma=1.0) # Very large volume
                else: # Rapid access/deletion or access to sensitive files
                    user_id = np.random.choice(common_users) # Could be common user
                    file_size = np.random.lognormal(mean=9, sigma=1.2) # Larger than usual
                    access_type = np.random.choice([2, 3], p=[0.7, 0.3]) # More deletions/copies
                    time_since_last_access = np.random.exponential(scale=10) # Very short time gaps
                    data_volume = np.random.lognormal(mean=7, sigma=0.8) # Larger volume

            current_sequence.append([user_id, file_size, access_type, time_since_last_access, data_volume])
        
        X.append(current_sequence)
        y.append(1 if is_anomaly_sequence else 0)

    return X, np.array(y)

print("Generating simulated data security logs...")
X_raw, y_labels = generate_simulated_security_data()
print(f"Generated {len(X_raw)} sequences.")
print(f"Number of anomalies: {np.sum(y_labels)}")

# --- 2. Data Preprocessing ---
# 2.1 Flatten and Collect all features for scaling and encoding
all_features = []
for seq in X_raw:
    all_features.extend(seq)
all_features_df = pd.DataFrame(all_features, columns=['user_id', 'file_size', 'access_type', 'time_since_last_access', 'data_volume'])

# 2.2 Label Encoding for categorical features (user_id, access_type)
# We need to fit encoders on the *entire* dataset to ensure consistent mapping.
user_id_encoder = LabelEncoder()
all_features_df['user_id_encoded'] = user_id_encoder.fit_transform(all_features_df['user_id'])

access_type_encoder = LabelEncoder()
all_features_df['access_type_encoded'] = access_type_encoder.fit_transform(all_features_df['access_type'])

# 2.3 Scaling for numerical features (file_size, time_since_last_access, data_volume)
# Use MinMaxScaler for numerical features.
scaler_file_size = MinMaxScaler()
all_features_df['file_size_scaled'] = scaler_file_size.fit_transform(all_features_df[['file_size']])

scaler_time = MinMaxScaler()
all_features_df['time_since_last_access_scaled'] = scaler_time.fit_transform(all_features_df[['time_since_last_access']])

scaler_data_volume = MinMaxScaler()
all_features_df['data_volume_scaled'] = scaler_data_volume.fit_transform(all_features_df[['data_volume']])

# 2.4 Reconstruct scaled sequences
X_processed = []
current_idx = 0
for seq in X_raw:
    processed_seq = []
    for _ in seq:
        # Extract processed features from the all_features_df based on current_idx
        processed_seq.append([
            all_features_df.loc[current_idx, 'user_id_encoded'],
            all_features_df.loc[current_idx, 'file_size_scaled'],
            all_features_df.loc[current_idx, 'access_type_encoded'],
            all_features_df.loc[current_idx, 'time_since_last_access_scaled'],
            all_features_df.loc[current_idx, 'data_volume_scaled']
        ])
        current_idx += 1
    X_processed.append(processed_seq)

# 2.5 Pad sequences to ensure uniform length for RNN input
# We use `padding='post'` and `value=0` (or a specific masking value)
# Masking layer in Keras can ignore these padded values during training.
max_sequence_length = max(len(seq) for seq in X_processed)
X_padded = pad_sequences(X_processed, maxlen=max_sequence_length, padding='post', dtype='float32', value=0.0)

# Determine number of features after processing
num_features = X_padded.shape[2]

print(f"Padded data shape: {X_padded.shape}")
print(f"Number of features per time step: {num_features}")
print("Data preprocessed and padded successfully.")

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 4. Define the RNN Model (LSTM) ---
# We use a Masking layer to ignore the padded values (0.0 in our case)
# so that the RNN doesn't learn from these artificial inputs.

model = Sequential([
    # Masking layer: It masks timesteps where all features are 0.0 (our padding value).
    # This tells the LSTM to ignore these padded timesteps.
    Masking(mask_value=0.0, input_shape=(max_sequence_length, num_features)),
    LSTM(units=128, activation='relu', return_sequences=True), # return_sequences=True for stacking LSTMs
    Dropout(0.3),
    LSTM(units=64, activation='relu'), # Last LSTM layer, return_sequences=False by default
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid') # Binary classification output
])

# --- 5. Compile the Model ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- 6. Train the Model ---
print("\nTraining the RNN model for data security threat detection...")
history = model.fit(
    X_train, y_train,
    epochs=25, # Increased epochs slightly
    batch_size=64,
    validation_split=0.15, # Use a portion of training data for validation
    verbose=1
)

print("\nModel training complete.")

# --- 7. Evaluate the Model ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 8. Make Predictions ---
num_predictions_to_show = 15
sample_indices = np.random.choice(len(X_test), num_predictions_to_show, replace=False)

sample_X = X_test[sample_indices]
sample_y_true = y_test[sample_indices]

predictions = model.predict(sample_X)
predicted_classes = (predictions > 0.5).astype(int) # Convert probabilities to binary classes

print("\n--- Predictions on Sample Test Data ---")
for i in range(num_predictions_to_show):
    true_label_str = "Anomaly" if sample_y_true[i] == 1 else "Normal"
    predicted_class_str = "Anomaly" if predicted_classes[i][0] == 1 else "Normal"
    match_status = " (MATCH)" if sample_y_true[i] == predicted_classes[i][0] else " (MISMATCH!)"
    print(f"Sample {i+1}: True Label = {true_label_str}, Predicted Prob = {predictions[i][0]:.4f}, Predicted Class = {predicted_class_str}{match_status}")


# --- 9. Plot Training History ---
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# --- Key Considerations for Real-World Data Security Threat Detection ---
# 1.  Data Source: Real-world data would come from SIEM (Security Information and Event Management)
#     systems, EDR (Endpoint Detection and Response) tools, DLP (Data Loss Prevention) logs,
#     file system audit logs, or cloud service logs (e.g., S3 access logs, Azure Blob storage logs).
# 2.  Feature Engineering: This is critical. Features should capture behavioral patterns:
#     -   Frequency of access to sensitive files
#     -   Unusual access times/locations
#     -   Changes in typical data transfer volumes for a user/system
#     -   Sequence of commands executed (e.g., `ls` -> `cp` -> `rm` on sensitive data)
#     -   Access to unusual file types or directories
# 3.  Time Windows and Sessionization: Events need to be grouped into meaningful sequences
#     or "sessions" based on time windows (e.g., 5-minute intervals) or logical user sessions.
# 4.  Unsupervised Learning: For true anomaly detection where you don't have labeled
#     "threat" data, autoencoders or one-class SVMs are often used. RNNs can be used
#     in an unsupervised way by training them to predict the next event in a sequence
#     and flagging events with high prediction error as anomalies.
# 5.  Alerting and Response: A detected anomaly needs to trigger an alert for security
#     analysts to investigate and potentially respond (e.g., block user, isolate system).
# 6.  Feedback Loop: Incorporate feedback from analysts to refine the model and reduce
#     false positives/negatives.
