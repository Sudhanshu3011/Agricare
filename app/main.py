from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib  # If you need sklearn to use this directly
from xgboost import XGBClassifier

# Load the models
rf_model = joblib.load("app/models/ml_model_1.pkl")
xgb_model = joblib.load("app/models/ml_model_2.pkl")
dl_model = tf.keras.models.load_model("app/models/dl_model.h5")

# FastAPI app instance
app = FastAPI()

# Pydantic schema to validate input data
class PredictionInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class PredictionOutput(BaseModel):
    model: str
    prediction: float


# Utility function to process the input
def preprocess_input(data):
    return np.array([[
        data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall
    ]])

# Prediction endpoint
@app.post("/predict/", response_model=PredictionOutput)
async def predict(data: PredictionInput, model_name: str = "rf"):
    """
    Predict the target using the chosen model (Random Forest, XGBoost, or DL).
    Accepts input data and model choice and returns predictions.
    """
    # Preprocess input data
    input_data = preprocess_input(data)

    # Model predictions based on the selected model
    if model_name == "rf":
        prediction = rf_model.predict(input_data)
        model_type = "Random Forest"
    elif model_name == "xgb":
        prediction = xgb_model.predict(input_data)
        model_type = "XGBoost"
    elif model_name == "dl":
        prediction = dl_model.predict(input_data)
        model_type = "Deep Learning"
    else:
        raise HTTPException(status_code=400, detail="Invalid model name. Choose from 'rf', 'xgb', or 'dl'.")

    return {"model": model_type, "prediction": prediction[0]}
