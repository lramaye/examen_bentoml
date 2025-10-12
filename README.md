# Examen BentoML

Ce repertoire contient l'architecture basique afin de rendre l'évaluation pour l'examen BentoML.

Vous êtes libres d'ajouter d'autres dossiers ou fichiers si vous jugez utile de le faire.

Voici comment est construit le dossier de rendu de l'examen:

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed      
│   │   └── raw           
│   ├── models      
│   ├── src       
│   └── README.md
```

Afin de pouvoir commencer le projet vous devez suivre les étapes suivantes:

- Forker le projet sur votre compte github

- Cloner le projet sur votre machine

- Récuperer le jeu de données à partir du lien suivant: [Lien de téléchargement]( https://datascientest.s3-eu-west-1.amazonaws.com/examen_bentoml/admissions.csv)

## Lancement

### environnement virtuel
```bash
virtualenv bentoml-env
source bentoml-env/bin/activate
pip install -r requirements.txt
```

### sequence
```bash
cd examen_bentoml
python3 src/prepare_data.py
python3 src/normalize.py
python3 src/grid_search.py
python3 src/train_model.py
```

### bentoml
