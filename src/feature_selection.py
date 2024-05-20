"""Module for feature selection methods."""

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mutual_info_score as MI
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import tqdm
from sklearn.model_selection import cross_val_score

from src.utils import CMI, interaction_gain


def CMIM(X, y, n_features=5):
    """Conditional Mutual Information Maximization."""
    selected = []
    for _ in range(n_features):
        max_cmim_value = float("-inf")
        for i in range(X.shape[1]):
            if i in selected:
                continue
            J = MI(X[:, i], y)
            max_value = float("-inf")
            for j in selected:
                curr_value = MI(X[:, i], X[:, j]) - CMI(X[:, i], X[:, j], y)
                if curr_value > max_value:
                    max_value = curr_value
            if J - max_value > max_cmim_value:
                max_cmim_value = J - max_value
                max_idx = i
        selected.append(max_idx)
    selected.sort()
    return selected


def JMIM(X, y, n_features=5):
    """Joint Mutual Information Maximization."""
    max_mi_value = float("-inf")
    for i in range(X.shape[1]):
        curr_mi = MI(X[:, i], y)
        if curr_mi > max_mi_value:
            first_idx = i
            max_mi_value = curr_mi
    selected = [first_idx]

    for _ in range(n_features - 1):
        max_jmim_value = float("-inf")
        for i in range(X.shape[1]):
            if i in selected:
                continue
            min_value = float("inf")
            for j in selected:
                curr_value = MI(X[:, j], y) + CMI(X[:, i], y, X[:, j])
                if curr_value < min_value:
                    min_value = curr_value
            if min_value > max_jmim_value:
                max_jmim_value = min_value
                max_idx = i
        selected.append(max_idx)
    selected.sort()
    return selected


def IGFS(X, y, n_features=5):
    """Information Gain Feature Selection."""
    selected = []
    for _ in range(n_features):
        max_igfs_value = float("-inf")
        for i in range(X.shape[1]):
            if i in selected:
                continue
            J = MI(X[:, i], y)
            inter_gain_sum = 0
            for j in selected:
                inter_gain_sum += interaction_gain(X[:, i], X[:, j], y)
            inter_gain_sum /= len(selected) + 1
            if J + inter_gain_sum > max_igfs_value:
                max_igfs_value = J + inter_gain_sum
                max_idx = i
        selected.append(max_idx)
    selected.sort()
    return selected


def wrapper_criterion(X, y, criterion="bic"):
    """Wrapper method for feature selection using OLS and AIC/BIC criterion."""
    selected = []
    best_score = None
    while True:
        if len(selected) == X.shape[1]:
            break
        min_score = float("inf")
        for i in range(X.shape[1]):
            if i in selected:
                continue
            model = sm.OLS(y, sm.add_constant(X[:, selected + [i]])).fit()
            if criterion == "bic":
                score_val = model.bic
            elif criterion == "aic":
                score_val = model.aic
            if score_val < min_score:
                min_score = score_val
                min_idx = i
        if best_score is None or min_score < best_score:
            selected.append(min_idx)
            best_score = min_score
        else:
            break
    selected.sort()
    return selected


def l1_selection(X, y):
    """L1 regularization for feature selection."""
    sfs_model = LogisticRegression(max_iter=1000, penalty="l1", solver="liblinear").fit(
        X, y
    )
    sfs_forward = SequentialFeatureSelector(
        sfs_model, n_features_to_select="auto", tol=0.001, direction="forward"
    ).fit(X, y)
    selected = list(np.argwhere(sfs_forward.get_support() * 1 > 0).T[0])
    selected.sort()
    return selected


def find_relevant_features(X, y):
    """Find relevant features using different feature selection methods."""
    relevant_features = {}
    relevant_features["BIC"] = wrapper_criterion(X, y, criterion="bic")
    relevant_features["AIC"] = wrapper_criterion(X, y, criterion="aic")
    relevant_features["CMIM"] = CMIM(X, y)
    relevant_features["JMIM"] = JMIM(X, y)
    relevant_features["IGFS"] = IGFS(X, y)
    relevant_features["L1"] = l1_selection(X, y)

    print("Calculations completed!")
    return relevant_features


def perform_feature_selection_on_all_datasets(data):
    """Perform feature selection on all datasets."""
    datasets = [key for key, _ in data.items()]
    relevant_features_all_datasets = {}

    for key in datasets:
        print(f"Finding relevant features for {key} dataset...")
        relevant_features_all_datasets[key] = find_relevant_features(
            data[key]["X_discr"], data[key]["y"]
        )

    return relevant_features_all_datasets


def evaluate_feature_selection(data, relevant_features, estimator, cv=5):
    """Evaluate feature selection methods using cross-validation."""
    accuracy_scores = {}
    for dataset, _ in relevant_features.items():
        accuracy_scores[dataset] = {}
        for method, val in tqdm(
            relevant_features[dataset].items(), f"Processing dataset {dataset}"
        ):
            X = data[dataset]["X_orig"][:, val]
            y = data[dataset]["y"]
            clf = estimator()
            accuracy_scores[dataset][method] = cross_val_score(clf, X, y, cv=3)

        clf = estimator()
        accuracy_scores[dataset]["full_data"] = cross_val_score(
            clf, data[dataset]["X_orig"], data[dataset]["y"], cv=cv
        )
    return accuracy_scores
