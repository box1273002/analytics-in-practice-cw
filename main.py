import pandas as pd
from numpy.typing import NDArray
import numpy as np


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates the bronze dataset, converting it into a
    consistent format suitable for feature engineering.
    """
    return bronze.copy()


def extract_features(
    silver: pd.DataFrame,
) -> NDArray[np.float64]:
    """
    Transforms the cleaned dataset, removing unnecessary info
    and converting columns into numeric features ready
    for training.
    """
    return np.asarray([[]])


def extract_labels(silver: pd.DataFrame) -> NDArray[np.float64]:
    """
    Extracts labels from the cleaned dataset.
    """
    return np.asarray([[]])


def train(x: NDArray, y: NDArray):
    """
    Trains a model based on the features and labels provided,
    returning the model.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch between the number of feature samples ({
                x.shape[0]
            }) and labels ({y.shape[0]})."
        )


if __name__ == "__main__":
    campaign_data = pd.read_csv("campaign_data.csv")
    print("===== Cleaning Data =====")
    cleaned_data = build_silver(campaign_data)
    print("===== Extracting Features =====")
    x = extract_features(cleaned_data)
    y = extract_labels(cleaned_data)
    print("===== Splitting Training and Testing Data =====")
    x_train, y_train = x, y
    print("===== Training Model =====")
    model = train(x_train, y_train)
