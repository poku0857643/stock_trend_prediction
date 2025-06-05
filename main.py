from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import torch
import joblib
import numpy as np
import os
from trendmodel.model import model_lstm

app = FastAPI()


####
# Constants
N_STEPS = 60
FEATURES_SCALER = ["open", "high", "low", "close", "volume"]
FEATURES_MODEL = ["open", "high", "low", "volume"]
TARGET_FEATURE = "close"
TARGET_INDEX = FEATURES_SCALER.index(TARGET_FEATURE)

# Load data
# mock_data = "/Users/eshan/PycharmProjects/FastAPIProject2/trendmodel/TSLA.csv"

# Load the model and scaler
# MODEL_PATH = os.getenv("MODEL_PATH", "model/model_lstm.pth")
# SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
MODEL_PATH = "trendmodel/model_checkpoints/latest_checkpoint.pth"

####

@app.get("/trend_predict")
def trend_predict(ticker: str = Query(..., description="Trend prediction ticker")):
    try:
        # Load data
        data_path = f"test_api/{ticker}.csv" # mock data path, fix later
        df = pd.read_csv(data_path)
        df = df[df["ticker"] == ticker.upper()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found or no data available.")

        # Load model and scaler
        model_path = MODEL_PATH
        scaler_path = f"trendmodel/save_scaler/scaler_{ticker}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file {model_path} not found.")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Scaler file {scaler_path} not found.")

        scaler = joblib.load(scaler_path)
        model = model_lstm.load_model_2(model_path, feature_dim =len(FEATURES_MODEL))

        # Predict the trend
        predicted_close = model_lstm.predict_next_day(
            df,
            FEATURES_SCALER,
            FEATURES_MODEL,
            model,
            scaler,
            TARGET_INDEX,
            n_steps=N_STEPS
        )
        if predicted_close is None:
            raise HTTPException(status_code=500, detail="Prediction failed.")
        return {
            "ticker": ticker.upper(),
            "predicted_next_close": predicted_close
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



