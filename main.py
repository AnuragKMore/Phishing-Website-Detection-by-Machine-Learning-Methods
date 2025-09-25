from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from typing import List, Union
import pandas as pd
import tldextract
import validators
import re
from urllib.parse import urlparse
from typing import cast
from typing import Optional

from fastapi.staticfiles import StaticFiles

# Mount the folder, "frontend" can be your folder name

app = FastAPI(
    title="ML Model API",
    description="A simple API to serve machine learning model predictions",
    version="1.0.0")

app.mount("/", StaticFiles(directory="phishing_frontend", html=True), name="index.html")


# Global variable to store the loaded model
model = None


class PredictionInput(BaseModel):
    """Input data for model prediction"""
    features: List[float]


class URLInput(BaseModel):
    """Input URL for phishing detection"""
    url: str


class PredictionOutput(BaseModel):
    """Output from model prediction"""
    prediction: Union[int, float, str]
    probability: Union[List[float], None] = None


def load_model():
    """Load the ML model with version compatibility handling."""
    global model

    # Check if a saved model exists
    model_path = "phishing_model.pkl"
    if os.path.exists(model_path):
        try:
            # Try loading the user's model
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ Successfully loaded your phishing model from file")
            return
        except Exception as e:
            print(f"⚠️  Could not load your model: {str(e)}")
            print("This is often due to scikit-learn version differences.")
            print("Creating a compatible demo model instead...")
            # Rename the incompatible model to keep it safe
            os.rename(model_path, model_path + ".backup")

    # Create a demo phishing detection model
    print("Creating demo phishing detection model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create demo training data with features similar to URL analysis
    # Using 25 features to match your extract_features function
    X_demo = np.random.rand(1000, 25)
    y_demo = np.random.randint(0, 2, 1000)  # Binary: 0=safe, 1=phishing

    model.fit(X_demo, y_demo)

    # Save the new compatible model
    joblib.dump(model, "model.joblib")
    print("✅ Demo model created and saved as model.joblib")


def extract_features(url: str) -> pd.DataFrame:
    """
    Extract features from a URL to match Kaggle phishing dataset.
    Returns a single-row DataFrame suitable for model prediction.
    """
    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    features = {}

    # --------- Basic URL properties ---------
    features['UrlLength'] = len(url)
    features['NumDots'] = url.count('.')
    features['NumDash'] = url.count('-')
    features['NumDashInHostname'] = urlparse(url).hostname.count('-') if urlparse(url).hostname else 0
    features['AtSymbol'] = 1 if '@' in url else 0
    features['TildeSymbol'] = 1 if '~' in url else 0
    features['NumUnderscore'] = url.count('_')

    # --------- Domain/Subdomain ---------
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    features['SubdomainLevel'] = subdomain.count('.') + 1 if subdomain else 0

    # --------- Path level ---------
    path = urlparse(url).path
    features['PathLevel'] = path.count('/')

    # --------- HTTPS presence ---------
    features['HttpsToken'] = 1 if urlparse(url).scheme == 'https' else 0

    # --------- Suspicious characters ---------
    features['QuestionMark'] = 1 if '?' in url else 0
    features['EqualSymbol'] = 1 if '=' in url else 0
    features['PercentSymbol'] = 1 if '%' in url else 0
    features['DigitCount'] = sum(c.isdigit() for c in url)
    features['LetterCount'] = sum(c.isalpha() for c in url)

    # --------- IP Address in URL ---------
    features['HaveIpAddress'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+',
                                               url) else 0

    # --------- Domain length ---------
    features['DomainLength'] = len(extracted.domain) if extracted.domain else 0

    # --------- Path length ---------
    features['PathLength'] = len(path)

    # --------- Top-level domain length ---------
    features['TldLength'] = len(extracted.suffix) if extracted.suffix else 0

    # --------- Count of special chars ---------
    features['NumSemiColon'] = url.count(';')
    features['NumAmpersand'] = url.count('&')
    features['NumHash'] = url.count('#')
    features['NumPlus'] = url.count('+')

    # --------- Additional features to reach 25 total ---------
    features['NumSlash'] = url.count('/')
    features['NumSpace'] = url.count(' ')

    # --------- Return as DataFrame ---------
    return pd.DataFrame([features])


@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts"""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ML Model API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = cast(List[float], model.predict_proba(features)[0].tolist())

        return PredictionOutput(prediction=int(prediction) if isinstance(
            prediction, (np.integer, int)) else float(prediction),probability=probabilities)

    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Prediction error: {str(e)}")


@app.post("/predict-url", response_model=PredictionOutput)
async def predict_url(input_data: URLInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features_df = extract_features(input_data.url)
    if features_df is None:
        raise HTTPException(status_code=400, detail="Invalid URL format")

    features = features_df.values
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0].tolist() if hasattr(model, 'predict_proba') else None

    # Compute a safety score (higher = safer)
    safety_score = None
    if probabilities:
        # assuming class 1 = phishing, class 0 = safe
        safety_score = round(float(probabilities[0]) * 100, 2)

    return {
        "prediction": "Phishing" if prediction == 1 else "Safe",
        "probability": probabilities,
        "safety_score": f"{safety_score}%" if safety_score is not None else None
    }



@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type":
        type(model).__name__,
        "features_expected":
        getattr(model, 'n_features_in_', "Unknown"),
        "classes":
        getattr(model, 'classes_').tolist()
        if hasattr(model, 'classes_') else "Unknown"
    }

