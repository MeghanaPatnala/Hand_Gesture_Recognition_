import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load gesture data
df = pd.read_csv("C:/Users/megha/PycharmProjects/hand_g/gesture_data.csv")
X = df.iloc[:, :-1]  # landmark coordinates
y = df.iloc[:, -1]   # gesture labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Accuracy check
acc = clf.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("🎯 Saved gesture_model.pkl")
