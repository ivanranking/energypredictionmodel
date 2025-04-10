from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import os

app = Flask(__name__)

# Directory to save uploaded files and the trained model
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, 'feature_names.pkl')

# Global variable to store the trained model and feature names
trained_model = None
feature_names = None

def save_model(model, feature_names):
    """Saves the trained model and feature names."""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)

def load_model():
    """Loads the trained model and feature names if they exist."""
    global trained_model
    global feature_names
    try:
        with open(MODEL_PATH, 'rb') as f:
            trained_model = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        return True
    except FileNotFoundError:
        return False

# Attempt to load the model on startup
load_model()

def train_model_function(data, target_column, feature_columns=None):
    """Trains a linear regression model."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")

    y = data[target_column]
    if feature_columns:
        if not all(col in data.columns for col in feature_columns):
            raise ValueError("One or more feature columns not found in the data.")
        X = data[feature_columns]
    else:
        X = data.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    return model, X.columns.tolist(), mse

# Knowledge Distillation (Simplified Example - Teacher is the same model)
def distill_knowledge(teacher_model, X_train, y_train, alpha=0.5, temperature=2.0):
    """
    A simplified example of knowledge distillation where the teacher and student
    are the same model. This demonstrates the concept of using soft targets.
    In a real scenario, the teacher would typically be a larger, more complex model.
    """
    student_model = LinearRegression()
    teacher_predictions = teacher_model.predict(X_train)

    # Create "soft targets" by combining true targets and teacher predictions
    soft_targets = (1 - alpha) * y_train + alpha * teacher_predictions

    # Train the student model on these soft targets (and potentially original data)
    student_model.fit(X_train, soft_targets)  # Or combine with original y_train

    return student_model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/train', methods=['POST'])
def train():
    if 'training_file' not in request.files:
        return render_template('index.html', error="No training file uploaded.", feature_names=feature_names)

    file = request.files['training_file']
    if file.filename == '':
        return render_template('index.html', error="No training file selected.", feature_names=feature_names)

    target_column = request.form.get('target_column')
    feature_columns_str = request.form.get('feature_columns')
    feature_columns = [col.strip() for col in feature_columns_str.split(',')] if feature_columns_str else None

    if not target_column:
        return render_template('index.html', error="Please specify the target column.", feature_names=feature_names)

    try:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        if filename.endswith('.csv'):
            data = pd.read_csv(filename)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(filename)
        else:
            return render_template('index.html', error="Unsupported file format. Please upload CSV or XLSX.", feature_names=feature_names)

        # Train the initial model (teacher in the distillation context)
        teacher_model, trained_features, mse = train_model_function(data, target_column, feature_columns)

        # Apply knowledge distillation (student is the same model for simplicity)
        X = data[trained_features]
        y = data[target_column]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        distilled_model = distill_knowledge(teacher_model, X_train, y_train, alpha=0.3, temperature=1.5) # Adjust alpha and temperature

        global trained_model
        global feature_names
        trained_model = distilled_model
        feature_names = trained_features
        save_model(trained_model, feature_names)

        return render_template('index.html', training_message=f"Model trained successfully with MSE: {mse:.2f}.", training_success=True, feature_names=feature_names)

    except ValueError as e:
        return render_template('index.html', error=str(e), feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during training: {str(e)}", feature_names=feature_names)

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if trained_model is None:
        return render_template('index.html', error="Model not trained yet. Please upload a training file and train the model.", feature_names=feature_names)

    if 'prediction_file' not in request.files:
        return render_template('index.html', error="No prediction file uploaded.", feature_names=feature_names)

    file = request.files['prediction_file']
    if file.filename == '':
        return render_template('index.html', error="No prediction file selected.", feature_names=feature_names)

    try:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        if filename.endswith('.csv'):
            data = pd.read_csv(filename)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(filename)
        else:
            return render_template('index.html', error="Unsupported file format for prediction. Please upload CSV or XLSX.", feature_names=feature_names)

        if not all(feature in data.columns for feature in feature_names):
            missing_features = [f for f in feature_names if f not in data.columns]
            return render_template('index.html', error=f"Prediction file missing required features: {', '.join(missing_features)}", feature_names=feature_names)

        predictions = trained_model.predict(data[feature_names])
        return render_template('index.html', prediction_file_message="Predictions generated successfully.", prediction_file_success=True, predictions=predictions.tolist(), feature_names=feature_names)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction from file: {str(e)}", feature_names=feature_names)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    if trained_model is None or feature_names is None:
        return render_template('index.html', error="Model not trained yet or feature names not loaded.", feature_names=feature_names)

    try:
        feature_values = [float(request.form.get(f'feature{i+1}')) for i in range(len(feature_names))]
        if len(feature_values) != len(feature_names):
            return render_template('index.html', error="Please provide values for all required features.", feature_names=feature_names)

        input_data = pd.DataFrame([feature_values], columns=feature_names)
        prediction = trained_model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f"Predicted energy consumption: {prediction:.2f}", feature_names=feature_names)

    except ValueError:
        return render_template('index.html', error="Invalid feature values entered. Please enter numeric values.", feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during manual prediction: {str(e)}", feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
