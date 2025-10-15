# Examen BentoML (Python 3.11 + BentoML 1.1.0)

Ce répertoire contient une application BentoML de régression Random Forest (admissions étudiants) avec authentification JWT et un pipeline de préparation/entraînement.

Cette version est vérifiée pour fonctionner avec:
- Python 3.11
- BentoML 1.1.11

Le fichier bentofile.yaml épingle BentoML 1.1.11 et fixe Python 3.11 pour l’image de conteneur BentoML.

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
- Installer BentoML CLI: sera installé via requirements.txt, sinon `pip install bentoml==1.2.20`
- Installer par prepare_data, sinon télécharger le dataset: https://datascientest.s3-eu-west-1.amazonaws.com/examen_bentoml/admissions.csv et placer le fichier dans `data/raw/`

## Créer l’environnement Python 3.11

Linux/macOS (bash/zsh):
```
cd ~/examen_bentoml
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Préparation des données et entraînement
Exécuter la séquence suivante depuis la racine du projet (environnement activé):
```
python3 src/prepare_data.py     # telecharge les données
python3 src/grid_search.py     # grid search utilisé par train_model : models/best_params.pkl
python3 src/train_model.py     # entraîne et enregistre le modèle BentoML "students_rf"
```
Vérifier que le modèle est bien dans le Model Store BentoML:
```
bentoml models list
```

## Construction du Bento et image Docker
Construire le Bento (archive exécutable) à partir du bentofile.yaml:
```
bentoml build --containerize
```
Lancer le conteneur:
Récuperer l'id de l'image et ajouter à la commande docker run en remplacant xxx par l'id
```
docker run --rm -d -p 3001:3000 LoicRamaye_AdmissionsPrediction:xxx
```
test rapide:
```
python src/test.py  # test la connexion au service
```

Pytest :
```
PYTHONPATH=src python -m pytest -v src/test_service.py --disable-warnings
```

## Sauvegarde de l'image docker 
```
docker save -o bento_image.tar LoicRamaye_AdmissionsPrediction
```

## Notes de compatibilité
- Python 3.11: les dépendances de requirements.txt sont compatibles (numpy, pandas, scikit-learn, PyJWT, etc.).
- BentoML 1.2.20: utilisé dans requirements.txt et épinglé dans bentofile.yaml
- Les chemins sont relatifs à la racine du projet; exécutez les scripts depuis cette racine.
- Emplacement de bentos : /home/ubuntu/bentoml/bentos

