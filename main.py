from datetime import time

from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import torch
import joblib
import numpy as np
import os
import aiohttp
import asyncio
from typing import Optional, Dict, List, Any
from pathlib import Path
from genstrategies import *
from sympy.codegen.ast import continue_
import httpx
from genstrategies.generator import Generator
from genstrategies.text_extractor import TextExtractor
from trendmodel.model import model_lstm
import logging
from pydantic import BaseModel
from trendmodel.model.model_lstm import predict_next_day
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
from dotenv import load_dotenv
app = FastAPI()
load_dotenv()


####
# Constants
N_STEPS = 60
FEATURES_SCALER = ["open", "high", "low", "close", "volume"]
FEATURES_MODEL = ["open", "high", "low", "volume"]
TARGET_FEATURE = "close"
TARGET_INDEX = FEATURES_SCALER.index(TARGET_FEATURE)

# Load data for trend prediction
# mock_data = "/Users/eshan/PycharmProjects/FastAPIProject2/trendmodel/TSLA.csv"

# Load the model and scaler
# MODEL_PATH = os.getenv("MODEL_PATH", "model/model_lstm.pth")
# SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
MODEL_PATH = "trendmodel/model_checkpoints/latest_checkpoint_0605.pth"

###
# configurations for genStratetgies
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTSOURCE_FOLDER_PATH = os.getenv("OUTSOURCE_FOLDER_PATH")
CLOUD_FOLDER_PATH = os.getenv("CLOUD_FOLDER_PATH")
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH")

####
class TrendPredictionRequest(BaseModel):
    tickers:List[str]
    confidence_threashold: float = 0.5

class StrategyGenerationRequest(BaseModel):
    local_folder: Optional[str] = None
    online_folder: Optional[str] = None
    cloud_folder: Optional[str] = None
    strategy_prompt: Optional[str] = None
    tickers: List[str] = ["AAPL"]
    use_internal_trend_api: bool = True
    external_api_url: Optional[str] = None
    api_headers: Optional[Dict] = None

class FaultTolerantTrendPredictor:
    def __init__(self):
        self.model_cache = {}
        self.scaler_cache = {}
        self.data_cache = {}
        self.cache_ttl = 300
        self.last_cache_time = {}

    def _is_cache_valid(self, key:str)-> bool:
        """Check if cached data is still valid"""
        if key not in self.last_cache_time:
            return False
        return time.time() - self.last_cache_time[key] < self.cache_ttl

    async def load_model_and_scaler(self, ticker: str) -> tuple:

        """Load model and scaler with caching and error handling"""
        cache_key = f"model_scaler_{ticker}"

        if self._is_cache_valid(cache_key) and cache_key in self.model_cache:
            return self.model_cache[cache_key], self.scaler_cache[cache_key]

        try:

            # Check if model file exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file {MODEL_PATH} not found.")
                raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

            # Check if scaler file exists
            SCALER_PATH = f"trendmodel/save_scaler/scaler_{ticker}.pkl"
            if not os.path.exists(SCALER_PATH):
                logger.warning(f"Specific scaler for {ticker} not found, trying default scaler.")
                # Try default scaler
                DEFAULT_SCALER_PATH =  f"trendmodel/save_scaler/default_scaler.pkl"
                if os.path.exists(DEFAULT_SCALER_PATH):
                    SCALER_PATH = DEFAULT_SCALER_PATH
                else:
                    raise FileNotFoundError(f"No scaler availablr for {ticker}")

            # Load model and scaler
            scaler = joblib.load(SCALER_PATH)
            model = model_lstm.load_model_2(MODEL_PATH, feature_dim=len(FEATURES_MODEL))

            # Cache the loaded model and scaler
            self.model_cache[cache_key] = model
            self.scaler_cache[cache_key] = scaler
            self.last_cache_time[cache_key] = time.time()

            logger.info(f"Successfully loaded model and scaler for {ticker}")
            return model, scaler
        except Exception as e:
            logger.error(f"Error loading model or scaler for {ticker}: {e}")
            raise

    async def load_data_with_fallback(self, ticker: str) -> pd.DataFrame:
        """Load data with multiple fallback strategies"""
        cache_key = f"data_{ticker}"

        # Check cache first
        if self._is_cache_valid(cache_key) and cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Strategy 1: Try specific ticker file
        data_paths = [
            f"test_data/csv_files/{ticker}.csv",
            f"test_api/{ticker}.csv",
            f"data/{ticker}.csv",
            f"market_data/{ticker}.csv",
            f"data/default_market_data.csv"
        ]

        for data_path in data_paths:
            try:
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)

                    # filter for specific ticker if column exists
                    if "ticker" in df.columns:
                        ticker_data = df[df["ticker"] == ticker.upper()]
                        if not ticker_data.empty:
                            self.data_cache[cache_key] = ticker_data
                            self.last_cache_time[cache_key] = time.time()
                            return ticker_data
                    else:
                        # Assume data is already for the specific ticker
                        self.data_cache[cache_key] = df
                        self.last_cache_time[cache_key] = time.time()
                        return df
            except Exception as e:
                logger.warning(f"Error loading {data_path}: {e}")
                continue
        raise HTTPException(status_code=404, detail=f"No data available for ticker {ticker}")

    async def predict_with_fallback(self, ticker: str) -> Dict[str, Any]:
        """Predict with comprehensive error handling and fallback"""
        try:
            # Load data
            df = await self.load_data_with_fallback(ticker)

            # Load model and scaler
            model, scaler = await self.load_model_and_scaler(ticker)

            # Ensure all required columns are present
            missing_cols = [col for col in FEATURES_SCALER if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for predictionL {missing_cols}")


            # Make prediction
            predicted_close = model_lstm.predict_next_day(
                df[FEATURES_SCALER],
                FEATURES_SCALER,
                FEATURES_MODEL,
                model,
                scaler,
                TARGET_INDEX,
                n_steps=N_STEPS
            )

            if predicted_close is None:
                raise ValueError("Prediction returned None")

            # Calculate confidence using the existing method
            confidence = self._calculate_confidence(df, predicted_close)

            return {
                "ticker": ticker.upper(),
                "predicted_next_close": float(predicted_close),
                "confidence": confidence,
                "data_points": len(df),
                "prediction_timestamp": time.time(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")

            # Return fallback prediction with low confidence
            return {
                "ticker": ticker.upper(),
                "predicted_next_close": 150.0, # default allback value
                "confidence": 0.1,
                "data_points": 0,
                "prediction_timestamp": time.time(),
                "status": "fallback",
                "error": str(e)
            }

    def _calculate_confidence(self, df: pd.DataFrame, prediction: float) -> float:
        """Calculate prediction confidence based on data quality"""
        try:
            data_points = len(df)
            if data_points > 30:
                return 0.3
            elif data_points < 100:
                return 0.6
            else:
                # Calculate based on recent volatility
                if 'close' in df.columns:
                    recent_std = df['close'].tail(20).std()
                    recent_mean = df['close'].tail(20).mean()
                    volatility_ratio = recent_std / recent_mean if recent_mean > 0 else 1

                    # Lower confidence for high volatility
                    if volatility_ratio > 0.1:
                        return 0.5
                    elif volatility_ratio > 0.05:
                        return 0.7
                    else:
                        return 0.8
                else:
                    return 0.6

        except Exception as e:
            print(f"Error calculating confidence for {e}")

# Global predictor instance
trend_predictor = FaultTolerantTrendPredictor()


@app.get("/trend_predict")
async def trend_predict(ticker: str = Query(..., description="Trend prediction ticker")):
    """Fault-tolerant trend prediction endpoint"""
    try:
        result = await trend_predictor.predict_with_fallback(ticker)

        if result['status'] == "fallback":
            # Return with warning status code but still provide data
            return {
                **result,
                "warning": "Prediction used fallback data due to errors"
            }
        return result
    except Exception as e:
        logger.error(f"Critical error in trend prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction service unvailable: {str(e)}")


@app.post("/trend_predict_batch")
async def trend_predict_batch(request: TrendPredictionRequest):
    """Batch trend prediction with fault tolerance"""
    predictions = []
    # Process predictions concurrently
    tasks = [trend_predictor.predict_with_fallback(ticker) for ticker in request.tickers]

    try:
        result = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(result):
            if isinstance(result, Exception):
                # Handle individual ticker failures
                predictions.append({
                    "ticker": request.tickers[i].upper(),
                    "predicted_next_close": 150.0,
                    "confidence": 0.1,
                    "status": "error",
                    "error": str(result)
                })
            else:
                predictions.append(result)

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    # Filtered_predictions by confidence threshold
    filtered_predictions = [
        p for p in predictions
        if p.get("confidence", 0) >= request.confidence_threshold
    ]

    return {
        "predictions": predictions,
        "filtered_predictions": filtered_predictions,
        "total_requested": len(request.tickers),
        "successful_predictions": len([p for p in predictions if p.get("status") == "success"]),
        "filtered_count": len(filtered_predictions)
    }

async def fetch_trend_predictions_internal(tickers: List[str], base_url: str = "http://localhost:8000") -> List[Dict]:
    """Fetch trend predictions from internal API"""
    predictions = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for ticker in tickers:
            try:
                response = await client.get(f"{base_url}/trend_predict", params= {"ticker": ticker})
                response.raise_for_status()
                predictions.append(response.json())
            except Exception as e:
                logger.error(f"Failed to fetch prediction for {ticker}: {e}")
                # Add fallback prediction
                predictions.append({
                    "ticker": ticker.upper(),
                    "predicted_next_close": 150.0,
                    "confidence": 0.1,
                    "status": "api_error"
                })
        return predictions

@app.post("/generate_strategies")
async def generate_strategies_integrated(request: StrategyGenerationRequest):
    """Integrated strategy generation with fault-tolerant trend prediction"""
    start_time = time.time()

    try:
        # Step 1. Get trend predictions
        if request.use_internal_trend_api:
            logger.info("Fetching trend predictions from internal API")
            trend_predictions = await fetch_trend_predictions_internal(request.tickers)
        elif request.external_api_url:
            logger.info("Fetching trend predictions from external API")
            # Use the fault-tolerant external APi fetcher from previous implementation
            trend_predictions = await trend_predictor.predict_with_fallback(
                primary_url=request.external_api_url,
                headers=request.api_headers or {}
            )
        else:
            logger.info("Using default trend predictions")
            trend_predictions = [{"prediction_close": 150.0, "confidence": 0.5}]

        # Step 2. Validate folder paths and extract PDFs
        folders = {
            "local": request.local_folder,
            "online": request.online_folder,
            "cloud": request.cloud_folder
        }

        embeddings = {}
        valid_folders = 0

        for folder_type, folder_path in folders.items():
            try:
                if folder_path and Path(folder_path).exists():
                    extractor = TextExtractor(folder_path)
                    embeddings[f"{folder_type}_embeddings"] = extractor.extract_text_from_pdfs()
                    valid_folders += 1
                    key = f"{folder_type}_embeddings"
                    logger.info(f"Successfullt processed {folder_type} folder: {len(embeddings[key])} documents")
                else:
                    embeddings[f"{folder_type}_embeddings"] = []
                    logger.warning(f"Folder {folder_type} not available: {folder_path}")
            except Exception as e:
                logger.error(f"Error processing {folder_type} folder: {e}")
                embeddings[f"{folder_type}_embeddings"] = []

        # Step 3: Validate we have sufficient data
        total_embeddings = sum(len(emb) for emb in embeddings.values())
        if total_embeddings == 0 and not trend_predictions:
            raise HTTPException(
                status_code=400,
                detail="No data available for strategy generation. Provide at least PDF folders or trend predictions."
            )

        # Step 4: Generate strategies
        try:
            generator = Generator(
                local_embeddings = embeddings.get("local_embeddings", []),
                outsource_embeddings = embeddings.get("outsource_embeddings", []),
                cloud_embeddings = embeddings.get("cloud_embeddings", []),
                trend_prediction = trend_predictions
            )

            if request.strategy_prompt:
                generator.set_strategy_prompt(request.strategy_prompt)

            strategies = generator.generate_strategies()
            return {
                "success": True,
                "strategies": strategies,
                "metadata": {
                    "trend_predictions": len(trend_predictions),
                    "successful_predictions": len([p for p in strategies if p.get("status") == "success"]),
                    "folders_processed": valid_folders,
                    "total_documents": total_embeddings,
                    "processing_time": time.time() - start_time,
                    "tickers_analyzed": request.tickers
                },
                "trend_data": trend_predictions
            }
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return {
                "success": False,
                "error": f"Strategy generation failed: {str(e)}",
                "partial_data": {
                    "trend_predictions_available": len(trend_predictions) > 0,
                    "documents_available": total_embeddings > 0,
                    "folders_processed": valid_folders
                },
                "processing_time": time.time() - start_time
            }

    except Exception as e:
        logger.error(f"Critical error in strategy generation: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy generation service unavailable: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "trend_prediction": "unknown",
            "model_loading": "unknown",
            "pdf_processing": "unknown"
        }
    }

    # Test trend prediction
    try:
        test_result = await trend_predictor.predict_with_fallback("AAPL")
        health_status["services"]["trend_prediction"] = "healthy" if test_result else "degraded"
    except:
        health_status["services"]["trend_prediction"] = "unhealthy"

    # PDF processing health check for local, online, and cloud folders
    pdf_folders = {
        "local": os.getenv("PDF_FOLDER_PATH", ""),
        "outsource": os.getenv("OUTSOURCE_FOLDER_PATH", ""),
        "cloud": os.getenv("CLOUD_FOLDER_PATH", ""),
    }
    pdf_statuses = {}
    for folder_type, folder_path in pdf_folders.items():
        try:
            if folder_path and Path(folder_path).exists() and any(Path(folder_path).glob("*.pdf")):
                extractor = TextExtractor(folder_path)
                _ = extractor.extract_text_from_pdfs()
                pdf_statuses[folder_type] = "healthy"
            else:
                pdf_statuses[folder_type]= "no_pdfs_found"
        except Exception as e:
            pdf_statuses[folder_type] = f"unhealthy ({str(e)})"

    health_status["services"]["pdf_processing"] = pdf_statuses

    # Test model loading
    try:
        if os.path.exists(MODEL_PATH):
            health_status["services"]["model_loading"] = "healthy"
        else:
            health_status["services"]["model_loading"] = "unhealthy"
    except:
        health_status["services"]["model_loading"] = "unhealthy"


    # Optionally, update the final status check:
    if any(v == "unhealthy" for v in health_status["services"].values()):
        health_status["status"] = "degraded"

    return health_status