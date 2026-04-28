from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("data/model/churn_model.pkl")

# Create FastAPI app
app = FastAPI()

# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerData):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # -----------------------------
    # Encode categorical columns
    # -----------------------------
    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    for col in categorical_columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_df)

    result = int(prediction[0])

    return {
        "prediction": result,
        "meaning": "Customer Will Churn" if result == 1 else "Customer Will Stay"
    }
