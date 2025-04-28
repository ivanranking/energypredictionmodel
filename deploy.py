from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def preprocess_data(df):
    # If TimeStamp exists, process it
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.sort_values('TimeStamp')
        
        df['Hour'] = df['TimeStamp'].dt.hour
        df['Minute'] = df['TimeStamp'].dt.minute
        df['DayOfWeek'] = df['TimeStamp'].dt.dayofweek
        
        # One-hot encode DayOfWeek
        encoder = OneHotEncoder(sparse_output=False)
        dayofweek_encoded = encoder.fit_transform(df[['DayOfWeek']])
        dayofweek_df = pd.DataFrame(dayofweek_encoded, columns=[f'DayOfWeek_{int(i)}' for i in range(dayofweek_encoded.shape[1])])
        
        df = pd.concat([df.reset_index(drop=True), dayofweek_df], axis=1)
        
        # Drop original TimeStamp and DayOfWeek columns
        df = df.drop(['TimeStamp', 'DayOfWeek'], axis=1)
    
    return df

def build_mlp(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn(input_shape):
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=(input_shape, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(input_shape, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    model_type = request.form['model_type']  # 'mlp', 'cnn', 'rnn', 'lstm'
    target_column = request.form['target_column']
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    df = pd.read_csv(filename)
    df = preprocess_data(df)
    
    if target_column not in df.columns:
        return jsonify({'error': 'Target column not found'}), 400
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'mlp':
        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    else:
        # Reshape for sequence models
        X_train_seq = np.expand_dims(X_train, axis=2)
        X_test_seq = np.expand_dims(X_test, axis=2)
        
        if model_type == 'cnn':
            model = build_cnn(X_train_seq.shape[1])
            model.fit(X_train_seq, y_train, epochs=20, batch_size=32, verbose=1)
        elif model_type == 'rnn':
            model = build_rnn(X_train_seq.shape[1])
            model.fit(X_train_seq, y_train, epochs=20, batch_size=32, verbose=1)
        elif model_type == 'lstm':
            model = build_lstm(X_train_seq.shape[1])
            model.fit(X_train_seq, y_train, epochs=20, batch_size=32, verbose=1)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

    model_save_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_model.h5")
    model.save(model_save_path)

    return jsonify({'message': f'Model trained and saved as {model_type}_model.h5'})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    model_name = request.form['model_name']  # e.g., 'mlp_model.h5'
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    df = pd.read_csv(filename)
    df = preprocess_data(df)

    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    model = load_model(model_path)

    if len(model.input_shape) == 2:
        X = df
    else:
        X = np.expand_dims(df, axis=2)

    predictions = model.predict(X).flatten()

    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port

