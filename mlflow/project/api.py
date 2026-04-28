from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("data/model/churn_model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MonthlyCharges: float
    TotalCharges: float

# Health check
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)

    result = int(prediction[0])

    return {
        "prediction": result,
        "meaning": "Customer Will Churn" if result == 1 else "Customer Will Stay"
    }
