import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("../data/processed/cleaned.csv")

# Encode text columns
for column in df.columns:
    if df[column].dtype == "object":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))

# Target
target_column = "Churn"

X = df.drop(columns=[target_column])
y = df[target_column]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load model
model = joblib.load("model/churn_model.pkl")

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Save report
with open("reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

print("Evaluation completed")
