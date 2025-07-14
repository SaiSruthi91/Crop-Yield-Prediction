# 🌾 Crop Yield Prediction using AI & ML

**End-to-End Crop Prediction based on Environmental and Agricultural Inputs**

---

## 🧠 Project overview

This project is aimed at predicting the **most suitable crop** for cultivation based on various factors such as rainfall, fertilizers, pesticides, crop year, and geographical state. Using machine learning models, specifically **Random Forest Classifier**, it helps farmers or agriculturalists make informed decisions.

The web app is built with **Flask (backend)** and has a clean **HTML/CSS-based frontend** that allows users to input data and receive predictions with visual feedback.

---

## ⚙️ Technologies Used

### 🔙 Backend

- Python 3.8+
- Flask
- Scikit-learn
- Pandas & NumPy
- Joblib (Model Persistence)

### 📊 ML Concepts

- Label Encoding
- Feature Engineering
- Train-Test Split
- Random Forest Classification
- Accuracy Evaluation

### 🌐 Frontend

- HTML5
- CSS3
- Bootstrap
- Jinja2

---

## 📁 Project Structure

```bash
crop-yield-prediction/
├── app.py                    # Main Flask application
├── model/
│   ├── crop_model.pkl       # Trained Random Forest model
│   ├── crop_data.csv        # Training dataset
│   ├── crop_encoder.pkl     # Label encoder for crop
│   ├── state_encoder.pkl    # Label encoder for state
│   └── season_encoder.pkl   # Label encoder for season
├── templates/
│   ├── index.html           # Input page
│   └── result.html          # Result page
├── static/
│   ├── images/              # Crop image assets
│   └── uploads/             # Uploaded field images
├── requirements.txt         # Required libraries
└── README.md
## 📊 Dataset

The dataset used for this project was obtained from Kaggle:

🔗 [Crop Yield in Indian States Dataset – by Akshat Gupta](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset)

### Dataset Details:
- Format: CSV (tab-separated recommended)
- Columns include:
  - `Crop`
  - `Crop_Year`
  - `Season`
  - `State`
  - `Area`
  - `Production`
  - `Annual_Rainfall`
  - `Fertilizer`
  - `Pesticide`
  - `Yield`

- Suitable for multi-class classification problems involving agricultural analytics and predictive modeling.

#### How to use:

1. Download the dataset from the link above.
2. Rename the file as `crop_data.csv` (if needed).

### 4. Modify your training script (e.g., `app.py` or `train_model.py`) to load the dataset:

```python
df = pd.read_csv("model/crop_data.csv", sep='\t')
```

3. Create a `model/` folder in your project directory.
4. Place the dataset inside:
```bash
crop-yield-prediction/
└── model/
    └── crop_data.csv
## 🔍 Features

✅ Predicts crop based on multiple input features  
✅ Upload image of your field (optional)  
✅ Dynamic image display of predicted crop  
✅ Clean UI with form validations  
✅ Feature Engineering for improved model accuracy

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Srilasya112/crop-yield-prediction.git
cd crop-yield-prediction
```

### 2. set up virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask App

```bash
python app.py
```

Access the app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Training (Optional)

If you'd like to retrain the model on the same or updated dataset:

```python
from app import train_and_save_model
train_and_save_model()
```

This script will:

* Load and clean the data  
* Encode categorical features  
* Engineer new features for better accuracy  
* Train a Random Forest Classifier  
* Save the model using `joblib`

---
