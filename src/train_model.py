import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

rf_regressor = RandomForestRegressor(n_jobs=-1, random_state=42)

#--Train the model
rf_regressor.fit(X_train, y_train)

#--Test the model
rf_regressor.predict(X_test)

#--Get the model accuracy
accuracy = rf_regressor.score(X_test, y_test)

print(f"Model accuracy: {accuracy}")

# test the model on a single observation
test_data = X_test.iloc[[0]]
# print the actual label
print(f"Actual label: {y_test[0]}")
# print the predicted label
print(f"Predicted label: {rf_regressor.predict(test_data)[0]}")

# Enregistrer le modèle dans le Model Store de BentoML
model_ref = bentoml.sklearn.save_model("accidents_rf", rf_regressor)

print(f"Modèle enregistré sous : {model_ref}")