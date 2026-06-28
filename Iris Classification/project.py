import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Load Dataset
# -------------------------------
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset Shape:", X.shape)
print()

# -------------------------------
# Data Visualization
# -------------------------------
sns.pairplot(pd.concat([X, pd.Series(y, name="Species")], axis=1), hue="Species")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Hyperparameter Tuning
# -------------------------------
parameters = {
    "n_estimators":[50,100,150],
    "max_depth":[2,4,6,None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    parameters,
    cv=5
)

grid.fit(X_train,y_train)

print("Best Parameters:",grid.best_params_)

model = grid.best_estimator_

# -------------------------------
# Cross Validation
# -------------------------------
scores = cross_val_score(model,X,y,cv=5)

print("Cross Validation Scores:",scores)
print("Average CV Accuracy:",scores.mean())

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

print()
print("Accuracy:",accuracy_score(y_test,y_pred))

print()
print("Classification Report")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(model,"iris_model.pkl")

print("Model Saved Successfully")

# -------------------------------
# Predict New Flower
# -------------------------------
sample = [[5.1,3.5,1.4,0.2]]

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print()
print("Predicted Flower:",iris.target_names[prediction[0]])
print("Confidence:",round(max(probability[0])*100,2),"%")



# Create DataFrame with predictions
result = X_test.copy()
result["Actual Species"] = [iris.target_names[i] for i in y_test]
result["Predicted Species"] = [iris.target_names[i] for i in y_pred]

# Save to CSV
result.to_csv("iris_predictions.csv", index=False)

print("iris_predictions.csv created successfully!")