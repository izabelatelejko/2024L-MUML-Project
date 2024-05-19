"""Module for generating synthetic data."""

import numpy as np
import pandas as pd


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
        X3 = X1 + np.random.normal(0, 0.1, (n, n_rel))
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
