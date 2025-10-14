import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import bentoml

# Chargement des données d'entraînement et de test
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

# Chargement de grid_search
#params_path = os.path.join("..", "..", "models", "best_params.pkl")
params_path = "models/best_params.pkl"
if not os.path.exists(params_path):
    raise FileNotFoundError(
        f"Fichier introuvable: {params_path}. "
    )

with open(params_path, "rb") as f:
    best_params = pickle.load(f)

# random_state
if "random_state" not in best_params:
    best_params["random_state"] = 42

model = RandomForestRegressor(**best_params)

# Fit du modèle
model.fit(X_train, y_train)

# Éval sur le jeu de test
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
# Get the model accuracy
accuracy = model.score(X_test, y_test)

print(f"Model accuracy: {accuracy}")
print("Meilleurs hyperparamètres :", best_params)
print("MSE :", mse_test)

# Sauvegarde du meilleur modèle entraîné
#output_dir = os.path.join("..", "..", "models", "models")
#output_dir = "models"
#os.makedirs(output_dir, exist_ok=True)
#model_path = os.path.join(output_dir, "best_model.pkl")
#with open(model_path, "wb") as f:
#    pickle.dump(model, f)

#print(f"Modèle sauvegardé dans : {model_path}")

# Enregistrer le modèle dans le Model Store de BentoML avec le tag "latest"
model_ref = bentoml.sklearn.save_model("students_rf:latest", model)

print(f"Modèle enregistré sous : {model_ref}")
