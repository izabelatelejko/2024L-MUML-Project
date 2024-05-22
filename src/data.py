"""Module for generating synthetic data."""

import numpy as np
import pandas as pd
from scipy.io import loadmat


def generate_data(
    n=1000,
    n_rel=5,
    n_irrel=30,
    betas=[1, 2, 3, 4, 5, 6],
    n_classes=10,
    dataset_variant=0,
):
    """Method generates synthetic data and target variable.

    X1 are relevant features - first n_rel columns in X
    X2, X3, and X4 are irrelevant features.

    Args:
        n: number of rows to generate
        n_rel: number of relevant features that will be generated from normal distribution
        n_iirel: number of irrelevant features that will be generated from normal distribution
        betas: list of coefficients for calculating target variable (must be of len n_rel + 1)
            the first element is bias
        n_classes: number of classes in target variable, classes are distributed evenly
        dataset_variant: defines which type of irrelevant features will be included in dataset:
                        - 0: include irrelevant features (y is not created based on them),
                        - 1: include copy of relevant features with added gaussian noise,
                        - 2: include interactions between relevant variables.

    Returns:
        X: synthetic data
        y: target variable
    """
    assert len(betas) == (n_rel + 1), "len of betas must be equal to (n_rel + 1)"
    assert dataset_variant in [
        0,
        1,
        2,
    ], "dataset variant must be an integer: 0, 1, or 2"

    # relevant features from normal distribution
    X1 = np.random.normal(0, 1, (n, n_rel))

    # irrelevant features from normal distribution
    if dataset_variant == 0:
        X2 = np.random.normal(0, 1, (n, n_irrel))
        X = np.concatenate([X1, X2], axis=1)
    # relevant features with noise
    elif dataset_variant == 1:
        X3 = X1 + np.random.normal(0, 0.5, (n, n_rel))
        X = np.concatenate([X1, X3], axis=1)
    # second order interactions of relevant features
    else:
        X4 = np.empty((n, 0), float)
        for i in range(n_rel - 1):
            for j in range(i + 1, n_rel):
                X4 = np.append(X4, np.expand_dims(X1[:, i] * X1[:, j], 1), axis=1)
        X = np.concatenate([X1, X4], axis=1)

    # target variable
    y = X1 @ np.array(betas[1:]).T + betas[0]
    y = pd.qcut(y, n_classes, labels=False)

    return X, y


def discretize_dataset(X, bins=10):
    X_discr = np.copy(X)
    for i in range(X.shape[1]):
        X_discr[:, i] = pd.cut(X[:, i], bins=bins, labels=False)

    return X_discr


def load_preprocess_data():
    data = {}

    # Divorce dataset, source https://www.kaggle.com/datasets/rabieelkharoua/split-or-stay-divorce-predictor-dataset
    divorce = pd.read_csv("data/divorce.csv", sep=";")

    data["divorce"] = {}
    data["divorce"]["X_orig"] = divorce.drop("Class", axis=1).to_numpy()
    data["divorce"]["X_discr"] = divorce.drop("Class", axis=1).to_numpy()
    data["divorce"]["y"] = divorce["Class"].to_numpy()
    data["divorce"]["n_features"] = data["divorce"]["X_orig"].shape[1]

    # AIDS classification dataset, source: https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction
    aids = pd.read_csv("data/aids.csv")
    X_aids = aids.drop("infected", axis=1).to_numpy()

    data["aids"] = {}
    data["aids"]["X_orig"] = X_aids
    data["aids"]["X_discr"] = discretize_dataset(X_aids)
    data["aids"]["y"] = aids["infected"].to_numpy()
    data["aids"]["n_features"] = X_aids.shape[1]

    # LOL Diamond FF15 dataset, source: https://www.kaggle.com/datasets/jakejoeanderson/league-of-legends-diamond-matches-ff15
    lol = pd.read_csv("data/lol.csv")
    X_lol = lol.drop(["match_id", "blue_Win"], axis=1).to_numpy()

    data["lol"] = {}
    data["lol"]["X_orig"] = X_lol
    data["lol"]["X_discr"] = discretize_dataset(X_lol)
    data["lol"]["y"] = lol["blue_Win"].to_numpy()
    data["lol"]["n_features"] = X_lol.shape[1]

    # Cancer dataset, source: https://www.kaggle.com/datasets/erdemtaha/cancer-data
    cancer = pd.read_csv("data/cancer.csv")
    cancer.loc[cancer["diagnosis"] == "M", "diagnosis"] = 0
    cancer.loc[cancer["diagnosis"] == "B", "diagnosis"] = 1
    X_cancer = cancer.drop(["id", "diagnosis", "Unnamed: 32"], axis=1).to_numpy()

    data["cancer"] = {}
    data["cancer"]["X_orig"] = X_cancer
    data["cancer"]["X_discr"] = discretize_dataset(X_cancer)
    data["cancer"]["y"] = cancer["diagnosis"].to_numpy().astype(int)
    data["cancer"]["n_features"] = X_cancer.shape[1]

    # Gait classification, source: https://archive.ics.uci.edu/dataset/604/gait+classification
    gait = loadmat("data/gait.mat")

    X_gait = gait["X"]
    y_gait = gait["Y"].T[0]

    inds_to_del = []
    for i in range(X_gait.shape[0]):
        if np.sum(np.isnan(X_gait[i, :])) != 0:
            inds_to_del.append(i)
    X_gait = np.delete(X_gait, (inds_to_del), axis=0)
    y_gait = np.delete(y_gait, (inds_to_del), axis=0)

    data["gait"] = {}
    data["gait"]["X_orig"] = X_gait
    data["gait"]["X_discr"] = discretize_dataset(X_gait)
    data["gait"]["y"] = pd.cut(y_gait, 3, labels=False)
    data["gait"]["n_features"] = X_gait.shape[1]

    # Generated data

    for dataset_variant in range(3):
        X_gen, y_gen = generate_data(
            n_rel=5,
            betas=[0, 1, 1, 1, 1, 1],
            dataset_variant=dataset_variant,
            n_classes=2,
        )
        gen_dataset_type = "generated_" + str(dataset_variant)
        data[gen_dataset_type] = {}
        data[gen_dataset_type]["X_orig"] = X_gen
        data[gen_dataset_type]["X_discr"] = discretize_dataset(X_gen)
        data[gen_dataset_type]["y"] = y_gen
        data[gen_dataset_type]["n_features"] = X_gen.shape[1]
        data[gen_dataset_type]["n_relevant"] = 5

    X_irrelevant = np.random.choice(2, (1000, 10))
    X_relevant = np.random.choice(2, (1000, 3))
    X = np.concatenate((X_relevant, X_irrelevant), axis=1)

    Y = X_relevant[:, 0]
    for i in range(1, X_relevant.shape[1]):
        Y = 1 * np.logical_xor(Y, X_relevant[:, i])

    data["xor"] = {}
    data["xor"]["X_orig"] = X
    data["xor"]["X_discr"] = discretize_dataset(X)
    data["xor"]["y"] = Y
    data["xor"]["n_features"] = X.shape[1]
    data["xor"]["n_relevant"] = X_relevant.shape[1]

    return data
