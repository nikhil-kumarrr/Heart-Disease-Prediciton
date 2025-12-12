# Heart Disease Prediction System
A machine learning–powered web app built with Scikit-Learn + Streamlit, designed to predict the risk of heart disease based on medical parameters.
It uses a trained classification model along with feature scaling to provide accurate risk estimation with a clean medical-grade UI.

## 🚀 Features

* ✔️ ML model trained using Heart Disease Statlog Dataset
* ✔️ Inputs for 13+ medical attributes
* ✔️ Real-time prediction using .joblib model
* ✔️ Risk score displayed in percentage
* ✔️ Progress bar visualization for heart risk
* ✔️ Beautiful hospital-theme UI with smooth hover animations
* ✔️ Validations & clean UI sections

## 🧠 How It Works
### 1. Dataset
You used the Heart Disease Statlog dataset containing features like:
* Age
* Chest pain type
* Cholesterol
* Resting BP
* Maximum heart rate
* ECG
* ST depression
* Number of vessels
* Thalassemia

### 2. Data Preprocessing
* Cleaning missing values
* Converting categories to numerical
* Scaling features using StandardScaler
* Splitting into train/test

### 3. Machine Learning Model
* Model type: Classification (SVM / Logistic Regression / RandomForest)
* Saved as: heart_model.joblib
* Scaler saved as: scaler.joblib

### 4. Prediction Pipeline

When a user gives inputs →
Values → Convert categorical → Scaled via saved scaler → Model predicts → App shows:
* High Risk
* Low Risk
* Risk %
* Progress bar

## Tech Stack
* Python
* Streamlit
* NumPy
* Pandas
* Scikit-Learn
* Joblib\
* Matplotlib / Seaborn

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
```

### 3️⃣ Activate the environment
#### Windows:
```bash
venv\Scripts\activate
```

#### Mac/Linux:
```bash
source venv/bin/activate
```

### 4️⃣ Install required libraries
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

## 📁 Project Structure
```bash
│── app.py                     # Streamlit UI + prediction logic
│── Heart Disease Prediction.ipynb   # Model training notebook
│── Heart_disease_statlog.csv       # Dataset
│── heart_model.joblib              # Trained ML model
│── scaler.joblib                   # Feature scaler
│── requirements.txt
└── README.md
```

## Dataset Info :
- Source: Kaggle – https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog
- Contains health indicators like age, cholesterol, resting blood pressure, max heart rate, and more.
- Target column: `target` (0 = No Heart Disease, 1 = Heart Disease Present)

## 🌐 Live Demo
https://predictionheartdiseaseapp.streamlit.app/

## Screenshots 
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202025-12-13%20010015.png?raw=true)
![img alt](https://github.com/nikhil-kumarrr/images/blob/main/Screenshot%202025-12-13%20010044.png?raw=true)
