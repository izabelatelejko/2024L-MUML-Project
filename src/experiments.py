"""Module for running experiments on the model."""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_val_score

from src.feature_selection import wrapper_criterion, CMIM, JMIM, IGFS, l1_selection


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


def evaluate_feature_selection(data, relevant_features, estimator, cv=5, **kwargs):
    """Evaluate feature selection methods using cross-validation."""
    accuracy_scores = {}
    for dataset, _ in relevant_features.items():
        accuracy_scores[dataset] = {}
        for method, val in tqdm(
            relevant_features[dataset].items(), f"Processing dataset {dataset}"
        ):
            X = data[dataset]["X_orig"][:, val]
            y = data[dataset]["y"]
            clf = estimator(**kwargs)
            accuracy_scores[dataset][method] = cross_val_score(clf, X, y, cv=3)

        clf = estimator()
        accuracy_scores[dataset]["full_data"] = cross_val_score(
            clf, data[dataset]["X_orig"], data[dataset]["y"], cv=cv
        )
    return accuracy_scores


def process_generated_data_results(data, relevant_features_all_datasets):
    """Process results for generated data."""
    features_comparison = pd.DataFrame()
    for dataset, _ in data.items():
        for method, val in relevant_features_all_datasets[dataset].items():
            row = {
                "dataset": dataset,
                "method": method,
                "n_selected": len(val),
                "n_relevant": data[dataset].get("n_relevant", None),
                "n_features": data[dataset]["n_features"],
            }
            features_comparison = pd.concat([features_comparison, pd.DataFrame([row])])
    features_comparison["selected_ratio"] = round(
        features_comparison["n_selected"] / features_comparison["n_features"], 2
    )
    return features_comparison.reset_index(drop=True)
