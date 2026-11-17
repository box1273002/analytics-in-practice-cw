import pandas as pd
from numpy._typing import NDArray
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


campaign_data = pd.read_csv("campaign_data.csv")
cleaned_data = build_silver(campaign_data)
x, y = extract_features(cleaned_data), extract_labels(cleaned_data)
# TODO train test split
model = train(x, y)
