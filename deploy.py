from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle
import os
from io import StringIO
import openpyxl

app = Flask(__name__)
app.secret_key = "energy_prediction_secret"  # Required for session

# Directory to save uploaded files and the trained model
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_FOLDER, 'feature_names.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_FOLDER, 'preprocessor.pkl')

# Global variable to store the trained model and feature names
trained_model = None
feature_names = None
preprocessor = None

# Mapping of model types to their classes
MODEL_TYPES = {
    'linear': LinearRegression,
    'sgd': SGDRegressor,
    'ridge': Ridge,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'mlp': MLPRegressor,
}

def save_model_artifacts(model, feature_names, preprocessor=None):
    """
    Saves the trained model, feature names, and preprocessor (if applicable) to files.
    """
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)
    if preprocessor:
        with open(PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(preprocessor, f)

def load_model_artifacts():
    """
    Loads the trained model, feature names, and preprocessor from files.
    Sets the global variables.
    """
    global trained_model
    global feature_names
    global preprocessor
    try:
        with open(MODEL_PATH, 'rb') as f:
            trained_model = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        return True
    except FileNotFoundError:
        return False

# Attempt to load model artifacts on startup
load_model_artifacts()

def preprocess_data(df, for_training=True):
    """
    Preprocesses the input dataframe.
    Handles potential missing 'TimeStamp' column and various data formats.

    Args:
        df (pd.DataFrame): The input dataframe.
        for_training (bool, optional):  If True, the function is being used for training.
            This affects how missing values in 'TimeStamp' are handled. Defaults to True.
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    if 'TimeStamp' not in df.columns:
        if for_training:
            raise ValueError("The training data does not contain a 'TimeStamp' column for the required preprocessing.")
        else:
            df['TimeStamp'] = pd.to_datetime('now')  # Create a dummy TimeStamp for prediction
    try:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    except ValueError:
        try:
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
        except ValueError:
            raise ValueError("Invalid TimeStamp format.  Please use '%m/%d/%Y %H:%M:%S' or a standard datetime format.")

    df = df.dropna(subset=['TimeStamp'])
    df['Hour'] = df['TimeStamp'].dt.hour
    df['Minute'] = df['TimeStamp'].dt.minute
    df['DayOfWeek'] = df['TimeStamp'].dt.dayofweek
    df = df.drop(columns=['TimeStamp'])
    df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DayOfWeek')
    return df

def train_model_function(data, target_column, model_type='linear', polynomial_degree=1):
    """
    Trains a regression model on the given data.

    Args:
        data (pd.DataFrame): The training data.
        target_column (str): The name of the target column.
        model_type (str, optional): The type of model to train.
            Defaults to 'linear'.  Options are defined in MODEL_TYPES.
        polynomial_degree (int, optional): The degree of the polynomial features. Defaults to 1.

    Returns:
        tuple: (trained model, feature names, preprocessor, MSE, R^2)
            Returns None if the model type is invalid.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")

    y = data[target_column]
    X = data.drop(columns=[target_column])

    numerical_features = ['Hour', 'Minute']
    numerical_cols_present = [col for col in numerical_features if col in X.columns]

    # Create a pipeline for preprocessing and model training
    pipeline_steps = []

    if numerical_cols_present:
        preprocessor = StandardScaler()
        pipeline_steps.append(('scaler', preprocessor))

    if polynomial_degree > 1:
        poly = PolynomialFeatures(degree=polynomial_degree)
        pipeline_steps.append(('poly', poly))

    model_class = MODEL_TYPES.get(model_type)
    if model_class:
        model = model_class()
        pipeline_steps.append(('model', model))
        pipeline = Pipeline(pipeline_steps)
    else:
        raise ValueError(f"Invalid model type: {model_type}.  Choose from {', '.join(MODEL_TYPES.keys())}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return pipeline, X.columns.tolist(), preprocessor, mse, r2

def distill_knowledge(teacher_model, X_train, y_train, student_model_type='linear', alpha=0.5, temperature=2.0):
    """
    Distills knowledge from a teacher model to a student model.

    Args:
        teacher_model: The trained teacher model.
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The target values for the training data.
        student_model_type (str, optional): The type of student model to use. Defaults to 'linear'.
        alpha (float, optional): Weighting between ground truth and soft targets. Defaults to 0.5.
        temperature (float, optional): Temperature for softening the teacher's predictions. Defaults to 2.0.

    Returns:
        The trained student model, or None on error
    """
    student_model_class = MODEL_TYPES.get(student_model_type)
    if not student_model_class:
        raise ValueError(f"Invalid student model type: {student_model_type}.  Choose from {', '.join(MODEL_TYPES.keys())}")
    student_model = student_model_class()

    teacher_predictions = teacher_model.predict(X_train)
    soft_targets = (1 - alpha) * y_train + alpha * teacher_predictions
    student_model.fit(X_train, soft_targets)
    return student_model

@app.route('/', methods=['GET'])
def index():
    """
    Renders the main page.
    Passes feature names to the template if available.
    """
    return render_template('index.html', feature_names=feature_names)

@app.route('/train', methods=['POST'])
def train():
    """
    Handles the model training process.
    -   Validates file upload and format (CSV, XLSX).
    -   Preprocesses the data.
    -   Trains the specified model.
    -   Distills knowledge into a student model.
    -   Saves the trained model and feature names.
    -   Returns the training results (MSE) or an error message.
    """
    global trained_model
    global feature_names
    global preprocessor

    if 'training_file' not in request.files:
        return render_template('index.html', error="No training file uploaded.", feature_names=feature_names)

    file = request.files['training_file']
    if file.filename == '':
        return render_template('index.html', error="No training file selected.", feature_names=feature_names)

    target_column = request.form.get('target_column')
    model_type = request.form.get('model_type', 'linear')  # Default to linear if not provided
    polynomial_degree = int(request.form.get('polynomial_degree', 1)) # default to 1

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

        data = preprocess_data(data.copy(), for_training=True)

        if target_column not in data.columns:
            return render_template('index.html', error=f"Target column '{target_column}' not found after preprocessing.", feature_names=feature_names)

        y = data[target_column]
        X = data.drop(columns=[target_column])

        teacher_model, trained_features, trained_preprocessor, mse, r2 = train_model_function(data.copy(), target_column, model_type, polynomial_degree)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        distilled_model = distill_knowledge(teacher_model, X_train, y_train, student_model_type=model_type, alpha=0.3, temperature=1.5)

        trained_model = distilled_model  # Use the distilled model
        feature_names = trained_features
        preprocessor = trained_preprocessor
        save_model_artifacts(trained_model, feature_names, preprocessor)
        session['trained_model_type'] = model_type #save model type

        return render_template('index.html',
                               training_message=f"Model ({model_type}) trained successfully with MSE: {mse:.2f}, R^2: {r2:.2f}.",
                               training_success=True,
                               feature_names=feature_names)

    except ValueError as e:
        return render_template('index.html', error=str(e), feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during training: {str(e)}", feature_names=feature_names)

@app.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Handles predictions from a file upload.
    -   Validates that a model has been trained.
    -   Reads the uploaded file (CSV, XLSX).
    -   Preprocesses the data.
    -   Makes predictions using the trained model.
    -   Returns the predictions or an error message.
    """
    if trained_model is None or feature_names is None or preprocessor is None:
        return render_template('index.html', error="Model not trained yet or artifacts not loaded.", feature_names=feature_names)

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

        data = preprocess_data(data.copy(), for_training=False)

        if not all(feature in data.columns for feature in feature_names):
            missing_features = [f for f in feature_names if f not in data.columns]
            return render_template('index.html', error=f"Prediction file missing required features after preprocessing: {', '.join(missing_features)}", feature_names=feature_names)

        predictions = trained_model.predict(data[feature_names])
        return render_template('index.html', prediction_file_message="Predictions generated successfully.", prediction_file_success=True, predictions=predictions.tolist(), feature_names=feature_names)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction from file: {str(e)}", feature_names=feature_names)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """
    Handles manual predictions from user-entered data.
    -   Validates that a model has been trained.
    -   Extracts input data from the form.
    -   Preprocesses the input data.
    -   Makes a prediction using the trained model.
    -   Returns the prediction or an error message.
    """
    if trained_model is None or feature_names is None or preprocessor is None:
        return render_template('index.html', error="Model not trained yet.", feature_names=feature_names)

    try:
        manual_data = {}
        manual_data['Hour'] = [int(request.form.get('manual-appliance1'))]
        manual_data['Minute'] = [int(request.form.get('manual-appliance2'))]
        day_of_week = int(request.form.get('manual-appliance3'))  # Get day of week
        manual_data['DayOfWeek'] = [day_of_week]

        input_df = pd.DataFrame(manual_data)
        input_df = preprocess_data(input_df.copy(), for_training=False)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        if not all(feature in input_df.columns for feature in feature_names):
            missing_manual_features = [f for f in feature_names if f not in input_df.columns]
            return render_template('index.html', error=f"Please provide values for all required features: {', '.join(missing_manual_features)}", feature_names=feature_names)
        prediction = trained_model.predict(input_df[feature_names])[0]
        return render_template('index.html', prediction_text=f"Predicted energy consumption: {prediction:.2f}", feature_names=feature_names)

    except ValueError:
        return render_template('index.html', error="Invalid feature values entered. Please enter numeric values.", feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during manual prediction: {str(e)}", feature_names=feature_names)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
