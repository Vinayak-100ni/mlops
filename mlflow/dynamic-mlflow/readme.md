# Dynamic Reusable MLflow Tracking Module for ML Developers

This approach is used in real projects where ML developers do not write MLflow code repeatedly inside every training script.

Instead, create a generic MLflow utility that developers can import and use dynamically.

---

# Goal

Instead of writing this repeatedly:

```python
mlflow.log_params()
mlflow.log_metrics()
mlflow.log_model()
```

Create one reusable file that handles everything.

---

# Recommended Project Structure

```text
mlops-project/
│
├── src/
│   ├── train.py
│   ├── utils/
│   │   └── mlflow_helper.py
│
├── data/
├── models/
├── config/
│   └── mlflow_config.yaml
└── requirements.txt
```

---

# Step 1 — Create Generic MLflow Helper

File:

```text
src/utils/mlflow_helper.py
```

```python
import mlflow
import mlflow.sklearn
import os


class MLflowHelper:

    def __init__(self, experiment_name, tracking_uri=None):

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def track(
        self,
        model,
        params,
        metrics,
        model_name="model",
        artifacts=None,
        tags=None,
        signature=None,
        input_example=None,
        run_name=None
    ):

        with mlflow.start_run(run_name=run_name):

            # Log Parameters
            if params:
                mlflow.log_params(params)

            # Log Metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # Log Tags
            if tags:
                mlflow.set_tags(tags)

            # Log Model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )

            # Log Artifacts
            if artifacts:
                for artifact in artifacts:
                    if os.path.exists(artifact):
                        mlflow.log_artifact(artifact)

            print("MLflow tracking completed")
```

---

# Step 2 — Use It Dynamically in Any Model Code

File:

```text
src/train.py
```

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils.mlflow_helper import MLflowHelper

# ----------------------
# Load Dataset
# ----------------------
df = pd.read_csv("../data/processed/cleaned.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ----------------------
# Train/Test Split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------
# Parameters
# ----------------------
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "random_state": 42
}

# ----------------------
# Model
# ----------------------
model = RandomForestClassifier(**params)

# Train Model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# ----------------------
# Metrics
# ----------------------
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "f1_score": f1_score(y_test, predictions)
}

# ----------------------
# MLflow Dynamic Tracking
# ----------------------
tracker = MLflowHelper(
    experiment_name="Customer Churn",
    tracking_uri="http://127.0.0.1:5000"
)

tracker.track(
    model=model,
    params=params,
    metrics=metrics,
    model_name="random_forest_model",
    run_name="RF_V1",
    tags={
        "developer": "ML Team",
        "project": "Churn Prediction"
    },
    artifacts=[
        "../reports/model_report.txt"
    ]
)
```

---

# What ML Developer Needs To Change

Only these things:

```python
params = {}
metrics = {}
model = AnyModel()
```

Everything else remains reusable.

---

# Example With XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(**params)
```

MLflow code remains SAME.

---

# Dynamic Usage Flow

```text
Developer Trains Model
        ↓
Gets Params + Metrics
        ↓
Calls tracker.track()
        ↓
MLflow Automatically Logs Everything
        ↓
Visible in MLflow UI
```
