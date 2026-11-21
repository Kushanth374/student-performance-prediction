import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("student_data.csv")
X = data.drop("result", axis=1)
y = data["result"]
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained successfully!")