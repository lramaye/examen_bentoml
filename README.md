# Examen BentoML (Python 3.11 + BentoML 1.1.0)

Ce répertoire contient une application BentoML de régression Random Forest (admissions étudiants) avec authentification JWT et un pipeline de préparation/entraînement.

Cette version est vérifiée pour fonctionner avec:
- Python 3.11
- BentoML 1.4.26

Le fichier bentofile.yaml épingle BentoML 1.4.26 et fixe Python 3.11 pour l’image de conteneur BentoML.

## Structure du projet
```
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── src/
│   ├── prepare_data.py
│   ├── normalize.py
│   ├── grid_search.py
│   ├── train_model.py
│   ├── service.py
│   └── test.py
├── bentofile.yaml
├── requirements.txt
└── README.md
```

## Pré-requis
- Installer Python 3.11
- Installer Docker (pour containeriser)
- Installer BentoML CLI: sera installé via requirements.txt, sinon `pip install bentoml==1.4.26`
- Télécharger le dataset: https://datascientest.s3-eu-west-1.amazonaws.com/examen_bentoml/admissions.csv et placer le fichier dans `data/raw/`

## Créer l’environnement Python 3.11

Windows (PowerShell):
```
cd C:\git\examen_bentoml
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
pip install -U pip
pip install -r requirements.txt
```

Linux/macOS (bash/zsh):
```
cd ~/examen_bentoml
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Préparation des données et entraînement
Exécuter la séquence suivante depuis la racine du projet (environnement activé):
```
python3 src/prepare_data.py
python3 src/grid_search.py     # écrit models/best_params.pkl
python3 src/train_model.py     # entraîne et enregistre le modèle BentoML "students_rf"
```
Vérifier que le modèle est bien dans le Model Store BentoML:
```
bentoml models list
```

## Servir le service localement (sans Docker)
Lancer le service BentoML directement depuis le code source:
```
bentoml serve src.service:rf_service --reload
```
Par défaut, l’API écoute sur http://127.0.0.1:3000

Tester les endpoints avec le script fourni (authentification + prédiction):
```
python src/test.py
```
Ce script:
- appelle POST /login avec des identifiants de démonstration (user123/password123)
- récupère un token JWT
- appelle POST /v1/models/rf_regressor/predict avec le header Authorization: Bearer <token>

## Construction du Bento et image Docker
Construire le Bento (archive exécutable) à partir du bentofile.yaml:
```
bentoml build
```
Containeriser le Bento en image Docker:
```
bentoml containerize rf_clf_service:latest
```
Lancer le conteneur:
```
docker run --rm -p 3000:3000 rf_clf_service:latest
```
Puis tester à nouveau:
```
python src/test.py
```

## Notes de compatibilité
- Python 3.11: les dépendances de requirements.txt sont compatibles (numpy, pandas, scikit-learn, PyJWT, etc.).
- BentoML 1.4.26: utilisé dans requirements.txt et épinglé dans bentofile.yaml. Le service utilise un Runner sklearn et des endpoints JSON compatibles 1.4.x.
- Les chemins sont relatifs à la racine du projet; exécutez les scripts depuis cette racine.