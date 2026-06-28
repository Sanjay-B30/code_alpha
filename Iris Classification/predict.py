import joblib
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

iris = load_iris()

sample = [[6.5,3.0,5.5,2.0]]

prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction[0]])