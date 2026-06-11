from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from joblib import load

app = FastAPI(title="NeXus Estate API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model
model = load("Nexus_Estate.joblib")

# Input schema — matches the 13 Boston Housing features
class HouseFeatures(BaseModel):
    CRIM: float       # Per capita crime rate
    ZN: float         # Proportion of residential land
    INDUS: float      # Non-retail business acres
    CHAS: float       # Charles River dummy (0 or 1)
    NOX: float        # Nitric oxide concentration
    RM: float         # Average rooms per dwelling
    AGE: float        # % owner-occupied units built before 1940
    DIS: float        # Distance to employment centres
    RAD: float        # Accessibility to highways
    TAX: float        # Property tax rate
    PTRATIO: float    # Pupil-teacher ratio
    B: float          # Proportion of Black residents
    LSTAT: float      # % lower status population

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[
        features.CRIM, features.ZN, features.INDUS, features.CHAS,
        features.NOX, features.RM, features.AGE, features.DIS,
        features.RAD, features.TAX, features.PTRATIO, features.B,
        features.LSTAT
    ]])
    prediction = model.predict(data)[0]
    return {
        "predicted_price": round(float(prediction), 2),
        "unit": "USD (thousands)"
    }
