from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import os
import sys
import logging
from pydantic import BaseModel
import pickle

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our project
from app.schemas.prediction import PredictionInput, PredictionOutput

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
def get_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model
def load_model():
    try:
        model_path = os.path.join("models", "model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                return pickle.load(file)
        else:
            logger.warning(f"Model file not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

@app.get("/")
async def root():
    return {"message": "Welcome to the MLOps FastAPI Project"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "warning", "message": "API is running but model is not loaded"}
    return {"status": "ok", "message": "API is running with model loaded"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Here you would preprocess the input data and format it for your model
        # For example:
        # features = preprocess_input(input_data)
        # prediction = model.predict([features])[0]
        
        # For now, we'll return a placeholder
        return PredictionOutput(
            prediction="placeholder",
            probability=0.95,
            model_version="0.1.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)