# Water-quality-prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("water_potability.csv")

# Fill missing values
data.fillna(data.mean(), inplace=True)

# Split data
X = data.drop("Potability", axis=1)
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))