import os
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Chargement uniquement des données d'entraînement
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()

# Définition du modèle et de la grille d'hyperparamètres
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# Recherche des meilleurs hyperparamètres (sans entraînement final du modèle)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Meilleurs hyperparamètres :", best_params)
print("MSE (CV moyenne) :", -grid_search.best_score_)

# Préparation du dossier de sortie
#output_dir = os.path.join("..", "..", "models")
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Sauvegarde des meilleurs hyperparamètres uniquement
params_path = os.path.join(output_dir, "best_params.pkl")
with open(params_path, "wb") as f:
    pickle.dump(best_params, f)
