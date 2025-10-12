import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making scaled data set')

    # Lecture des fichiers issu de make_dataset
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # Initialisation de MinMaxScaler
    scaler = MinMaxScaler()

    # Normalisation de X_train et X_test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # sauvegarde des dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # création des fichiers de sortie
    X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)


if __name__ == '__main__':
    main()