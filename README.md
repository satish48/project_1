# 📱 Teen Phone Usage Predictor 📊

![Project Banner](https://user-images.githubusercontent.com/yourusername/yourproject/banner-image.png)

---

## Overview

**Teen Phone Usage Predictor** is a machine learning project aimed at predicting teen phone usage behavior based on demographic, behavioral, and psychological features. This project uses advanced regression models, including CatBoost, XGBoost, and Random Forest, to forecast metrics like daily usage hours, phone checks, social media time, and gaming time.

This predictor can help parents, educators, and researchers understand patterns and potentially promote healthier digital habits among teenagers.

---

## Features / Input Variables

- Age  
- Daily Usage Hours  
- Sleep Hours  
- Exercise Hours  
- Apps Used Daily  
- Time on Social Media  
- Time on Gaming  
- Weekend Usage Hours  

---

## Tech Stack

- Python 3.x  
- Flask (Web app framework)  
- Scikit-learn, CatBoost, XGBoost (Machine learning models)  
- Pandas, NumPy (Data handling)  
- Bootstrap 5 (Frontend styling)  
- Git & GitHub (Version control)  

---

## Project Structure

```plaintext
├── app.py                 # Flask application entry point  
├── artifacts/             # Saved models and preprocessor objects  
├── src/                   # Source code for ML pipeline and utilities  
│   ├── pipeline/          # Model training and prediction pipeline  
│   ├── components/        # Data ingestion, model trainer, etc.  
│   ├── exception.py       # Custom exception handling  
│   ├── logger.py          # Logging utilities  
│   └── utils.py           # Helper functions  
├── templates/             # HTML templates for Flask app  
│   └── home.html  
├── static/                # CSS, JS, images (if any)  
├── requirements.txt       # Python dependencies  
└── README.md              # Project documentation  

## How to Use
1. **Clone the repository:**

    ```bash
    git clone https://github.com/satish48/project_1.git
    cd project_1
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask app:**

    ```bash
    python app.py
    ```

5. **Open your browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

6. **Enter the required features in the form and get your prediction!**

---

Usage: 
--> Input the required values such as daily usage hours, phone checks per day, apps      used, time spent on social media, and gaming.


## Click Predict.

Get the predicted phone usage output instantly.
Model Performance
Model	R² Score
CatBoostRegressor	0.97
RandomForestRegressor	0.86
XGBRegressor	0.85
AdaBoostRegressor	0.69
Linear Regression	0.64
Decision Tree	0.56


Contact
Created by Satish Mudrakola
📧 Email: your_email@example.com
🔗 LinkedIn: linkedin.com/in/satishmudrakola