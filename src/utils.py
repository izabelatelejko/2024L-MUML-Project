"""Module for feature selection utils."""

import numpy as np
from sklearn.metrics import mutual_info_score as MI


def entropy(X):
    """Entropy of a given variable."""
    _, counts = np.unique(X, return_counts=True)
    prob = counts / len(X)
    return -np.sum(prob * np.log2(prob))


def CMI(X, Y, Z):
    """Conditional mutual information between X and Y given Z."""
    cmi = 0
    for z in np.unique(Z):
        cmi += MI(X[Z == z], Y[Z == z]) * (len(Z[Z == z]) / len(Z))
    return cmi


def interaction_gain(X1, X2, Y):
    """Interaction Gain between X1 and X2 given Y."""
    return CMI(X1, Y, X2) - MI(X1, Y)


def NMI(X, Y):
    """Normalized mutual information between two variables."""
    return MI(X, Y) / min(entropy(X), entropy(Y))


def MI_battiti(X, S, Y):
    """Conditional mutual information between X and Y given S proposed by Battiti."""
    if S.shape[1] == 0:
        print(NMI(X, Y))
        return NMI(X, Y)
    beta = 1 / S.shape[1]
    sum_mi = 0
    for i in range(S.shape[1]):
        sum_mi += NMI(X, S[:, i])
    return NMI(X, Y) - beta * sum_mi


def MI_kwak(X, S, Y):
    """Conditional mutual information between X and Y given S proposed by Kwak."""
    if S.shape[1] == 0:
        return MI(X, Y)
    beta = 1 / S.shape[1]
    sum_mi = 0
    for i in range(S.shape[1]):
        sum_mi += MI(Y, S[:, i]) * MI(X, S[:, i]) / entropy(S[:, i])
    return MI(X, Y) - beta * sum_mi
