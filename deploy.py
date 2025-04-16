from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
PREPROCESSOR_PATH = os.path.join(MODEL_FOLDER, 'preprocessor.pkl')

# Global variable to store the trained model and feature names
trained_model = None
feature_names = None
preprocessor = None

def save_model_artifacts(model, feature_names, preprocessor=None):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)
    if preprocessor:
        with open(PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump(preprocessor, f)

def load_model_artifacts():
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

def preprocess_data(df):
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['TimeStamp'])
    df['Hour'] = df['TimeStamp'].dt.hour
    df['Minute'] = df['TimeStamp'].dt.minute
    df['DayOfWeek'] = df['TimeStamp'].dt.dayofweek
    df = df.drop(columns=['TimeStamp'])
    df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DayOfWeek')
    return df

def train_model_function(data, target_column, feature_columns=None):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")

    y = data[target_column]
    X = data.drop(columns=[target_column])

    numerical_features = ['Hour', 'Minute']
    numerical_cols_present = [col for col in numerical_features if col in X.columns]

    preprocessor = StandardScaler()
    if numerical_cols_present:
        X_scaled = preprocessor.fit_transform(X[numerical_cols_present])
        X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols_present, index=X.index)
        X_processed = pd.concat([X_scaled_df, X.drop(columns=numerical_cols_present, errors='ignore')], axis=1)
    else:
        X_processed = X

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))

    return model, X_processed.columns.tolist(), preprocessor, mse

def distill_knowledge(teacher_model, X_train, y_train, alpha=0.5, temperature=2.0):
    student_model = LinearRegression()
    teacher_predictions = teacher_model.predict(X_train)
    soft_targets = (1 - alpha) * y_train + alpha * teacher_predictions
    student_model.fit(X_train, soft_targets)
    return student_model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/train', methods=['POST'])
def train():
    global trained_model
    global feature_names
    global preprocessor

    if 'training_file' not in request.files:
        return render_template('index.html', error="No training file uploaded.", feature_names=feature_names)

    file = request.files['training_file']
    if file.filename == '':
        return render_template('index.html', error="No training file selected.", feature_names=feature_names)

    target_column = request.form.get('target_column')

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

        if 'TimeStamp' in data.columns:
            data = preprocess_data(data.copy())
        else:
            return render_template('index.html', error="The training data does not contain a 'TimeStamp' column for the required preprocessing.", feature_names=feature_names)

        if target_column not in data.columns:
            return render_template('index.html', error=f"Target column '{target_column}' not found after preprocessing.", feature_names=feature_names)

        y = data[target_column]
        X = data.drop(columns=[target_column])

        teacher_model, trained_features, trained_preprocessor, mse = train_model_function(data.copy(), target_column)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        distilled_model = distill_knowledge(teacher_model, X_train, y_train, alpha=0.3, temperature=1.5)

        trained_model = distilled_model
        feature_names = trained_features
        preprocessor = trained_preprocessor
        save_model_artifacts(trained_model, feature_names, preprocessor)

        return render_template('index.html', training_message=f"Model trained successfully with MSE: {mse:.2f}.", training_success=True, feature_names=feature_names)

    except ValueError as e:
        return render_template('index.html', error=str(e), feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during training: {str(e)}", feature_names=feature_names)

@app.route('/predict_file', methods=['POST'])
def predict_file():
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

        if 'TimeStamp' in data.columns:
            data = preprocess_data(data.copy())
        else:
            return render_template('index.html', error="The prediction data does not contain a 'TimeStamp' column for the required preprocessing.", feature_names=feature_names)

        if not all(feature in data.columns for feature in feature_names):
            missing_features = [f for f in feature_names if f not in data.columns]
            return render_template('index.html', error=f"Prediction file missing required features after preprocessing: {', '.join(missing_features)}", feature_names=feature_names)

        numerical_features = ['Hour', 'Minute']
        numerical_cols_present = [col for col in numerical_features if col in data.columns]
        if numerical_cols_present:
            data[numerical_cols_present] = preprocessor.transform(data[numerical_cols_present])

        predictions = trained_model.predict(data[feature_names])
        return render_template('index.html', prediction_file_message="Predictions generated successfully.", prediction_file_success=True, predictions=predictions.tolist(), feature_names=feature_names)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction from file: {str(e)}", feature_names=feature_names)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    if trained_model is None or feature_names is None or preprocessor is None:
        return render_template('index.html', error="Model not trained yet or artifacts not loaded.", feature_names=feature_names)

    try:
        manual_data = {}
        manual_data['Hour'] = [int(request.form.get('feature1'))]
        manual_data['Minute'] = [int(request.form.get('feature2'))]
        day_of_week = int(request.form.get('feature3'))
        manual_data['DayOfWeek'] = [day_of_week]

        input_df = pd.DataFrame(manual_data)
        input_df = pd.get_dummies(input_df, columns=['DayOfWeek'], prefix='DayOfWeek')
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        numerical_features = ['Hour', 'Minute']
        numerical_cols_present = [col for col in numerical_features if col in input_df.columns]
        if numerical_cols_present:
            input_df[numerical_cols_present] = preprocessor.transform(input_df[numerical_cols_present])

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
    app.run(debug=True)
