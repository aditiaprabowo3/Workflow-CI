import pandas as pd
import mlflow
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("submission model")
mlflow.autolog()

n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])
dataset_path = sys.argv[3]

df = pd.read_csv(dataset_path)
X = df.drop(columns='pass_all')
y = df['pass_all']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Akurasi: {acc}")
mlflow.log_metric("accuracy", acc)
