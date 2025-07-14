from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import uuid
from sklearn.metrics import accuracy_score
import json

app = Flask(__name__)

# Paths
MODEL_PATH = 'model/crop_model.pkl'
DATA_PATH = 'model/crop_data.csv'
SEASON_STATE_PATH = 'model/valid_labels.json'

@app.route('/')
def home():
    return render_template('index.html')

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        return False

    df = pd.read_csv(DATA_PATH, sep='\t')
    df.columns = df.columns.str.strip()

    # Replace infinite values with NaN, then drop all rows with any NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({
        'Annual_Rainfall': df['Annual_Rainfall'].median(),
        'Fertilizer': df['Fertilizer'].median(),
        'Pesticide': df['Pesticide'].median()
    }, inplace=True)
    df.dropna(inplace=True)  # Ensures no NaNs remain

    df['Season'] = df['Season'].astype(str).str.strip()
    df['State'] = df['State'].astype(str).str.strip()
    df['Crop'] = df['Crop'].astype(str).str.strip()


    valid_labels = {
        "seasons": sorted(df['Season'].unique().tolist()),
        "states": sorted(df['State'].unique().tolist())
    }
    with open(SEASON_STATE_PATH, 'w') as f:
        json.dump(valid_labels, f)

    le_season = LabelEncoder()
    le_state = LabelEncoder()
    le_crop = LabelEncoder()

    df['Season'] = le_season.fit_transform(df['Season'])
    df['State'] = le_state.fit_transform(df['State'])
    df['Crop'] = le_crop.fit_transform(df['Crop'])

    joblib.dump(le_crop, 'model/crop_encoder.pkl')
    joblib.dump(le_season, 'model/season_encoder.pkl')
    joblib.dump(le_state, 'model/state_encoder.pkl')

    X = df.drop(['Crop'], axis=1)
    y = df['Crop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    return True

# Force retrain to generate valid_labels.json
train_and_save_model()

model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        for key in ['year', 'area', 'production', 'rainfall', 'fertilizer', 'pesticide', 'yield']:
            raw = data.get(key, "").replace(",", "").strip()
            if raw == "" or raw.lower() == "none":
                raise ValueError(f"Missing or invalid input for {key}")
            data[key] = float(raw)

        file = request.files.get('crop_image')
        uploaded_image_url = None
        if file and file.filename:
            filename = str(uuid.uuid4()) + "_" + file.filename
            upload_path = os.path.join("static/uploads", filename)
            file.save(upload_path)
            uploaded_image_url = url_for('static', filename='uploads/' + filename)

        input_data = pd.DataFrame([{
            'Crop_Year': int(data['year']),
            'Season': data['season'].strip(),
            'State': data['state'].strip(),
            'Area': data['area'],
            'Production': data['production'],
            'Annual_Rainfall': data['rainfall'],
            'Fertilizer': data['fertilizer'],
            'Pesticide': data['pesticide'],
            'Yield': data['yield']
        }])

        state_encoder = joblib.load('model/state_encoder.pkl')
        season_encoder = joblib.load('model/season_encoder.pkl')
        crop_encoder = joblib.load('model/crop_encoder.pkl')

        with open(SEASON_STATE_PATH) as f:
            valid_labels = json.load(f)

        if input_data['Season'].iloc[0] not in valid_labels['seasons']:
            return jsonify({'error': f"Unknown season: {input_data['Season'].iloc[0]}"}), 400
        if input_data['State'].iloc[0] not in valid_labels['states']:
            return jsonify({'error': f"Unknown state: {input_data['State'].iloc[0]}"}), 400

        input_data['Season'] = season_encoder.transform(input_data['Season'])
        input_data['State'] = state_encoder.transform(input_data['State'])

        features = ['Crop_Year', 'Season', 'State', 'Area', 'Production',
                    'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
        encoded_prediction = model.predict(input_data[features])[0]
        prediction = crop_encoder.inverse_transform([encoded_prediction])[0]

        if prediction.lower() == 'turmeric' and input_data['Area'].iloc[0] > 10000:
            prediction = 'Rice'

        crop_key = prediction.lower().replace(" ", "_").replace("-", "_")
        image_name_map = {
            "lady_finger": "ladysfinger",
            "soyabean": "soya_bean",
            "groundnut": "ground_nut",
            "sugarcane": "sugar_cane",
        }
        crop_key = image_name_map.get(crop_key, crop_key)
        image_filename = f'images/{crop_key}.jpg'
        image_path = os.path.join('static', image_filename)

        image_url = url_for('static', filename=image_filename if os.path.exists(image_path) else 'images/no_image.jpg')

        return render_template('result.html',
                               prediction=prediction,
                               input_data=data,
                               image_url=image_url,
                               uploaded_image_url=uploaded_image_url)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

def static_file_exists(filename):
    static_path = os.path.join(app.static_folder, filename)
    return os.path.isfile(static_path)

@app.context_processor
def utility_processor():
    return dict(static_files=static_file_exists)

if __name__ == '__main__':
    app.run(debug=True)
