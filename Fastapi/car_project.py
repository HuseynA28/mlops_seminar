import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Annotated, Literal
import mlflow
from contextlib import asynccontextmanager

import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
STAGE = os.getenv("MLFLOW_STAGE", "Production")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "UsedCarPricePredictor")

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    try:
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model {MODEL_NAME} ({STAGE}) loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        raise
    yield  
  


app = FastAPI(lifespan=lifespan)

# Predict endpoint (now uses pre-loaded model)
@app.put("/get_prediction/")
def get_prediction(
    miles: Annotated[int, Query(ge=0.0)] = 86132,
    year: Annotated[int, Query(ge=1886)] =2010,
    engine_size: Annotated[float, Query(ge=0.9)] = 1.5,
    make: Annotated[Literal['toyota', 'honda'], Query()] = 'toyota',
    model: Annotated[Literal[
        'Prius', 'Highlander', 'Civic', 'Accord', 'Corolla', 'Ridgeline',
        'Odyssey', 'CR-V', 'Pilot', 'Camry Solara', 'Matrix', 'RAV4',
        'Rav4', 'HR-V', 'Fit', 'Yaris', 'Yaris iA', 'Tacoma', 'Camry',
        'Avalon', 'Venza', 'Sienna', 'Passport', 'Accord Crosstour',
        'Crosstour', 'Element', 'Tundra', 'Sequoia', 'Corolla Hatchback',
        '4Runner', 'Echo', 'Tercel', 'MR2 Spyder', 'FJ Cruiser',
        'Corolla iM', 'C-HR', 'Civic Hatchback', '86', 'S2000', 'Supra',
        'Insight', 'Clarity', 'CR-Z', 'Prius Prime', 'Prius Plug-In',
        'Prius c', 'Prius C', 'Prius v'
    ], Query()] = 'Prius',
    state: Annotated[Literal[
        'NB', 'QC', 'BC', 'ON', 'AB', 'MB', 'SK', 'NS', 'PE', 'NL', 'YT', 'NC', 'OH', 'SC'
    ], Query(description="Choose predefined state")] = 'NB',
):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Bug fix: use 'model' (the param), not 'model_name'
    dummy_input = pd.DataFrame([{
        "miles": miles,
        "year": year,
        "engine_size": engine_size,
        "make": make,
        "model": model,  
        "state": state
    }])

    try:
        predicted_price = app.state.model.predict(dummy_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predicted_price": predicted_price.tolist()}