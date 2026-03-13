import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = []
labels = []

dataset_path = "dataset"

for file in os.listdir(dataset_path):
    
    label = file.split(".")[0]
    
    df = pd.read_csv(os.path.join(dataset_path, file), header=None)
    
    for row in df.values:
        data.append(row)
        labels.append(label)

X = pd.DataFrame(data)
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")