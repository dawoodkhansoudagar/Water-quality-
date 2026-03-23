# Water Quality Prediction using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (you can replace with your own CSV file)
data = {
    'pH': [7, 6.5, 8, 5.5, 7.5, 6, 8.5, 7.2],
    'Turbidity': [3, 5, 2, 8, 4, 7, 1, 3],
    'Dissolved_Oxygen': [8, 6, 9, 5, 7, 6, 10, 8],
    'Temperature': [25, 30, 22, 35, 28, 32, 20, 26],
    'Safe': [1, 0, 1, 0, 1, 0, 1, 1]  # 1 = Safe, 0 = Not Safe
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('Safe', axis=1)
y = df['Safe']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example prediction
sample = [[7.0, 3, 8, 25]]  # pH, Turbidity, DO, Temp
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Water is Safe")
else:
    print("Water is Not Safe")
