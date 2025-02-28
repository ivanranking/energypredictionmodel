import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.mean(x, dim=2)
        return self.fc(x).squeeze()

# Paths
MODEL_PATH = "cnn_model.pth"
SCALER_PATH = "scaler.pkl"

# Train model if not found
def train_and_save_model():
    print("Training CNN Model...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate random training data
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(1000, 10, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    model = CNNModel(input_channels=10, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor.squeeze())
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained and saved as cnn_model.pth")

if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load trained model and scaler
model = CNNModel(input_channels=10, output_dim=1)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
scaler = joblib.load(SCALER_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
        tensor_features = tensor_features.view(1, tensor_features.shape[1], 1)
        
        with torch.no_grad():
            prediction = model(tensor_features).item()
        
        return render_template("index.html", prediction_text=f"Predicted Energy Consumption: {prediction:.4f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host ='0.0.0.0',port=10000)
