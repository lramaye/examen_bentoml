import os
import logging
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder

def download_csv(url: str, raw_dir: str, filename: str = "admission.csv") -> str:
    # Utilisation du script check_structure
    if check_existing_folder(raw_dir):
        os.makedirs(raw_dir)

    output_path = os.path.join(raw_dir, filename)

    if not check_existing_file(output_path):
        return output_path

    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download file. HTTP {resp.status_code} from {url}")

    with open(output_path, "wb") as f:
        f.write(resp.content)

    return output_path


def prepare_dataset(raw_csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(raw_csv_path, sep=",", header=0)

    # renommage des colonnes car des espaces sont presents
    columns = [
        "Serial",
        "GRE_score",
        "TOEFL_score",
        "University_Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
        "Chance_of_Admit",
    ]
    if len(df.columns) != len(columns):
        # Attempt to coerce to the expected width; if not matching, raise a clear error
        raise ValueError(
            f"Unexpected number of columns: got {len(df.columns)}, expected {len(columns)}"
        )
    df.columns = columns

    # Suppression de la colonne qui n est pas utilisable
    if "Serial" in df.columns:
        df = df.drop(columns=["Serial"])

    return df


def split_and_save(df: pd.DataFrame, processed_dir: str, test_size: float = 0.2, random_state: int = 42) -> None:
    # separation jeu entrainement et jeu de tests
    target_col = None
    target_col = "Chance_of_Admit"

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save files
    outputs = {
        os.path.join(processed_dir, "X_train.csv"): X_train,
        os.path.join(processed_dir, "X_test.csv"): X_test,
        os.path.join(processed_dir, "y_train.csv"): y_train,
        os.path.join(processed_dir, "y_test.csv"): y_test,
    }

    for path, data in outputs.items():
        if check_existing_file(path):
            # Convert Series to DataFrame for consistent CSV shape if needed
            if isinstance(data, pd.Series):
                data.to_csv(path, index=False, header=True)
            else:
                data.to_csv(path, index=False)


def main(
    raw_dir: str = os.path.join("data", "raw"),
    processed_dir: str = os.path.join("data", "processed"),
    url: str = "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv"):

    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation...")

    raw_csv_path = download_csv(url=url, raw_dir=raw_dir)
    logger.info(f"Raw data saved to: {raw_csv_path}")

    df = prepare_dataset(raw_csv_path)

    split_and_save(df, processed_dir=processed_dir)
    logger.info(f"Processed datasets saved to: {processed_dir}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
