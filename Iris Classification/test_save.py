from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()

X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "iris_model.pkl")

print("Model saved successfully!")