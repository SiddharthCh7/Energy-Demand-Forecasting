from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from xgboost import XGBRegressor

# Load the model


app = FastAPI()
model = XGBRegressor()
model.load_model("xgboost_model.json")

# Define input schema
class InputData(BaseModel):
    features: list  # list of features

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
