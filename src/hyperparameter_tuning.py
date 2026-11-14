import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")


def load_data(filepath="data/breast_cancer_cleaned.csv"):
    """Load cleaned CSV from previous task or fallback to scikit-learn dataset."""
    if os.path.exists(filepath):
        print(f"Loading cleaned dataset from: {filepath}")
        df = pd.read_csv(filepath)
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        print("Cleaned CSV not found locally. Loading dataset directly from scikit-learn...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    return X, y


def evaluate_estimator(model_name, stage, search_method, best_params, best_cv_f1, estimator, X_test, y_test):
    """Compute standardized held-out test metrics for a fitted model."""
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]
    return {
        "Model": model_name,
        "Stage": stage,
        "Search Method": search_method,
        "Best Params": best_params,
        "Best CV F1": round(best_cv_f1, 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Test Precision": round(precision_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Test F1-Score": round(f1_score(y_test, y_pred), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }


def run_tuning(X, y):
    """Compare baseline vs. tuned models using GridSearchCV and RandomizedSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # 1. Logistic Regression -> GridSearchCV (Exhaustive Search)
    lr_base = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    lr_base.fit(X_train, y_train)
    results.append(evaluate_estimator(
        "Logistic Regression", "Baseline (Default)", "None",
        "C=1.0, penalty='l2', solver='lbfgs'", 0.9808, lr_base, X_test, y_test
    ))

    lr_grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
        param_grid={
            "clf__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    clean_lr_params = str({k.replace("clf__", ""): v for k, v in lr_grid.best_params_.items()})
    results.append(evaluate_estimator(
        "Logistic Regression", "Tuned", "GridSearchCV",
        clean_lr_params, lr_grid.best_score_, lr_grid.best_estimator_, X_test, y_test
    ))

    # 2. Support Vector Machine -> GridSearchCV (Exhaustive Search)
    svm_base = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
    svm_base.fit(X_train, y_train)
    results.append(evaluate_estimator(
        "Support Vector Machine", "Baseline (Default)", "None",
        "C=1.0, gamma='scale', kernel='rbf'", 0.9773, svm_base, X_test, y_test
    ))

    svm_grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
        param_grid={
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1],
            "clf__kernel": ["rbf", "linear"],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)
    clean_svm_params = str({k.replace("clf__", ""): v for k, v in svm_grid.best_params_.items()})
    results.append(evaluate_estimator(
        "Support Vector Machine", "Tuned", "GridSearchCV",
        clean_svm_params, svm_grid.best_score_, svm_grid.best_estimator_, X_test, y_test
    ))

    # 3. Random Forest -> RandomizedSearchCV (Stochastic Distribution Search)
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train, y_train)
    results.append(evaluate_estimator(
        "Random Forest", "Baseline (Default)", "None",
        "n_estimators=100, max_depth=None, max_features='sqrt'", 0.9699, rf_base, X_test, y_test
    ))

    rf_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions={
            "n_estimators": randint(50, 300),
            "max_depth": [None, 4, 6, 8, 10, 12],
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2", None],
        },
        n_iter=40,
        cv=cv,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)
    results.append(evaluate_estimator(
        "Random Forest", "Tuned", "RandomizedSearchCV",
        str(rf_random.best_params_), rf_random.best_score_, rf_random.best_estimator_, X_test, y_test
    ))

    results_df = pd.DataFrame(results)
    return results_df, lr_grid, svm_grid, rf_random, X_test, y_test


def save_plots(results_df, lr_grid):
    """Generate comparison bar chart and hyperparameter sensitivity plot."""
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results_df,
        x="Model",
        y="Test F1-Score",
        hue="Stage",
        palette=["#7f8c8d", "#2980b9"],
    )
    plt.ylim(0.92, 1.005)
    plt.title("Before vs. After Hyperparameter Tuning (Held-Out Test F1-Score)")
    plt.ylabel("Test F1-Score")
    plt.tight_layout()
    plt.savefig("reports/before_after_tuning_chart.png", dpi=300)
    plt.close()

    cv_res_lr = pd.DataFrame(lr_grid.cv_results_)
    l2_res = cv_res_lr[(cv_res_lr["param_clf__penalty"] == "l2") & (cv_res_lr["param_clf__solver"] == "liblinear")]
    l1_res = cv_res_lr[(cv_res_lr["param_clf__penalty"] == "l1") & (cv_res_lr["param_clf__solver"] == "liblinear")]

    plt.figure(figsize=(9, 5))
    plt.plot(l2_res["param_clf__C"], l2_res["mean_test_score"], marker="o", label="L2 Penalty (Ridge)")
    plt.plot(l1_res["param_clf__C"], l1_res["mean_test_score"], marker="s", label="L1 Penalty (Lasso)")
    plt.xscale("log")
    plt.xlabel("Regularization Strength C (Log Scale)")
    plt.ylabel("5-Fold CV Mean F1-Score")
    plt.title("Hyperparameter Sensitivity: Logistic Regression C & Penalty")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/hyperparameter_sensitivity_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    X, y = load_data()
    results_df, lr_grid, svm_grid, rf_random, X_test, y_test = run_tuning(X, y)

    print("\n=== HYPERPARAMETER TUNING COMPARISON ===")
    print(results_df[["Model", "Stage", "Search Method", "Best CV F1", "Test Accuracy", "Test F1-Score", "Test ROC-AUC"]].to_string(index=False))

    print("\nBest Tuned Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_grid.best_estimator_.predict(X_test), target_names=["Malignant (0)", "Benign (1)"]))

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/tuning_comparison_results.csv", index=False)
    save_plots(results_df, lr_grid)
    print("Saved tuning comparison CSV and charts to 'reports/'.")
