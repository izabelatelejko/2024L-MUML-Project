"""Module for feature selection utils."""

import numpy as np
from sklearn.metrics import mutual_info_score as MI


def CMI(X, Y, Z):
    """Conditional mutual information between X and Y given Z."""
    cmi = 0
    for z in np.unique(Z):
        cmi += MI(X[Z == z], Y[Z == z]) * (len(Z[Z == z]) / len(Z))
    return cmi


def interaction_gain(X1, X2, Y):
    """Interaction Gain between X1 and X2 given Y."""
    return CMI(X1, Y, X2) - MI(X1, Y)
