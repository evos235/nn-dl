import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = "/content/sample_data/heart_disease.csv"  # Update this with your dataset's file path
data = pd.read_csv(file_path)

# Assuming the target column is named "target" and features are numerical
X = data.drop(columns=["target"]).values
y = data["target"].values

# Preprocess the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  # Convert to one-hot encoding for classification

# Preprocess the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for RNN input (RNN expects a 3D input: samples, timesteps, features)
X = np.expand_dims(X, axis=1)  # Add a time dimension

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')  # Adjust for the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("heart_disease_rnn_model.h5")
