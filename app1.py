import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Directory and file paths
PREDICTIONS_DIR = 'static'
PREDICTIONS_FILE = 'new_predictions.csv'
PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, PREDICTIONS_FILE)

def normalize_column_names(df):
    """Normalize column names to match expected names."""
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
    return df

@app.route('/')
def index():
    if os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        result_html = df.to_html(classes='table table-striped', index=False)
        return render_template('results.html', tables=[result_html], titles=[''])
    else:
        return render_template('no_predictions.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename.endswith(('.txt', '.csv')):
            return jsonify({'error': 'Invalid file type. Only TXT and CSV files are accepted.'}), 400

        # Read the file into a DataFrame
        df = pd.read_csv(file, delimiter=',')  # Adjust delimiter if needed
        original_df = df.copy()  # Keep a copy of the original DataFrame

        # Normalize column names
        df = normalize_column_names(df)

        # Drop unnecessary columns
        columns_to_drop = ['sensor_15', 'Unnamed: 0', 'Id']  # Add more columns if needed
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Check if 'Timestamp' column exists
        if 'Timestamp' not in df.columns:
            return jsonify({'error': "'Timestamp' column is missing from the input data."}), 400

        # Convert 'Timestamp' column to datetime format
        #df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

        # Handle missing values
        df.dropna(subset=['Timestamp'], inplace=True)

        # Feature Engineering
        df['year'] = df['Timestamp'].dt.year
        df['month'] = df['Timestamp'].dt.month
        df['day'] = df['Timestamp'].dt.day
        df['hour'] = df['Timestamp'].dt.hour
        df['minute'] = df['Timestamp'].dt.minute
        df['second'] = df['Timestamp'].dt.second
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df = df.drop(columns=['Timestamp'])

        # Ensure the DataFrame contains the expected features
        expected_features = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03',
                             'sensor_04', 'sensor_05', 'sensor_06', 'sensor_07',
                             'sensor_08', 'sensor_09', 'sensor_10', 'sensor_11',
                             'sensor_12', 'sensor_13', 'sensor_14', 'sensor_16',
                             'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
                             'sensor_21', 'sensor_22', 'sensor_23', 'sensor_24',
                             'sensor_25', 'sensor_26', 'sensor_27', 'sensor_28',
                             'sensor_29', 'sensor_30', 'sensor_31', 'sensor_32',
                             'sensor_33', 'sensor_34', 'sensor_35', 'sensor_36',
                             'sensor_37', 'sensor_38', 'sensor_39', 'sensor_40',
                             'sensor_41', 'sensor_42', 'sensor_43', 'sensor_44',
                             'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48',
                             'sensor_49', 'sensor_50', 'sensor_51', 'year',
                             'month', 'day', 'hour', 'minute', 'second', 
                             'hour_sin', 'hour_cos']

        df = df[[col for col in expected_features if col in df.columns]]

        # Normalize the features
        X_scaled = scaler.transform(df)

        # Make predictions
        y_pred = model.predict(X_scaled)

        # Map numerical labels to class names
        y_pred_class_names = label_encoder.inverse_transform(y_pred)

        # Create a result DataFrame with only the Timestamp from the original DataFrame and predictions
        result_df = original_df.loc[df.index, ['Timestamp']].copy()
        result_df['Prediction'] = y_pred_class_names

        result_json = result_df.to_dict(orient='records')
        
        # Save the predictions to a CSV file
        if not os.path.exists(PREDICTIONS_DIR):
            os.makedirs(PREDICTIONS_DIR)
        result_df.to_csv(PREDICTIONS_PATH, index=False)

        # Convert the DataFrame to HTML for rendering
        result_html = result_df.to_html(classes='table table-striped', index=False)

        # Render the results in the HTML page
        return render_template('results.html', tables=[result_html], titles=[''])

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/latest_predictions')
def latest_predictions():
    if os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        result_html = df.to_html(classes='table table-striped', index=False)
        return jsonify({'html': result_html})
    else:
        return jsonify({'error': 'No predictions available.'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)
