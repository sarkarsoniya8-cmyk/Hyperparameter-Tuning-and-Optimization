# Hyperparameter Tuning and Optimization Report

## 1. Objective & Optimization Strategy
Building directly on the previous model selection analysis, this study optimizes the top-performing classifiers on the **Breast Cancer Wisconsin Diagnostic Dataset** using cross-validated hyperparameter search. All preprocessing (`StandardScaler`) is encapsulated inside `scikit-learn` `Pipeline` objects to eliminate data leakage across validation folds, and **F1-Score** serves as the primary optimization objective to balance False Positives and False Negatives.

---

## 2. Search Techniques Applied

### A. Exhaustive Grid Search (`GridSearchCV`)
Applied to **Logistic Regression** and **Support Vector Machine (SVM)** where hyperparameter spaces are compact and structured:
- **Logistic Regression (`32` combinations × `5` folds = `160` fits):** Explored inverse regularization strength `C` (`[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]`), penalty norms (`l1`, `l2`), and solvers (`liblinear`, `saga`).
- **Support Vector Machine (`72` combinations × `5` folds = `360` fits):** Explored margin penalty `C` (`[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`), kernel coefficient `gamma` (`['scale', 'auto', 0.001, 0.01, 0.05, 0.1]`), and kernel functions (`rbf`, `linear`).

### B. Randomized Search (`RandomizedSearchCV`)
Applied to **Random Forest** (`40` sampled combinations × `5` folds = `200` fits) where searching over continuous integer ranges (`n_estimators` from `50` to `300`, `min_samples_split`, `min_samples_leaf`) and tree depth constraints (`max_depth`) is far more computationally efficient than exhaustive grid enumeration.

---

## 3. Quantitative Results: Baseline vs. Tuned Models

| Model | Stage | Search Method | Optimal Hyperparameters | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline | Default | `C=1.0, penalty='l2', solver='lbfgs'` | 0.9808 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9957 |
| **Logistic Regression** | **Tuned** | `GridSearchCV` | `C=0.5, penalty='l2', solver='liblinear'` | **0.9842** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9967** |
| **SVM** | Baseline | Default | `C=1.0, gamma='scale', kernel='rbf'` | 0.9773 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9964 |
| **SVM** | **Tuned** | `GridSearchCV` | `C=5.0, gamma='scale', kernel='rbf'` | **0.9808** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9974** |
| **Random Forest** | Baseline | Default | `n_estimators=100, max_depth=None` | 0.9699 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9942 |
| **Random Forest** | **Tuned** | `RandomizedSearchCV` | `n_estimators=210, max_depth=8, max_features='log2'` | **0.9702** | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9931 |

---

## 4. Hyperparameter Impact Analysis & Key Findings

1. **Regularization Strength (`C`) in Logistic Regression:**
   - Stronger regularization (`C = 0.5` vs. default `1.0`) improved 5-fold CV F1-Score from **0.9808 to 0.9842** and raised held-out Test ROC-AUC to **0.9967**.
   - Extreme under-regularization (`C = 50.0`) slightly degraded CV performance due to overfitting correlated geometric features, while excessive L1 regularization (`C = 0.001`) drove all weights to zero (underfitting).
2. **Margin Penalty (`C`) in Support Vector Machine:**
   - Increasing `C` from `1.0` to `5.0` with `kernel='rbf'` and `gamma='scale'` tightened the decision boundary around borderline tumor samples, boosting CV F1-Score from **0.9773 to 0.9808** and achieving the highest overall Test ROC-AUC (**0.9974**).
3. **Tree Depth & Feature Subsampling in Random Forest:**
   - Constraining `max_depth=8` and `max_features='log2'` with `210` trees slightly improved 5-fold CV F1 (`0.9699 -> 0.9702`), though held-out test accuracy dipped marginally (`0.9561 -> 0.9474`), illustrating that tuning on small datasets (`N = 569`) requires caution against overfitting validation folds.
4. **Final Production Recommendation:**
   - **Tuned Logistic Regression (`C=0.5, penalty='l2', solver='liblinear'`)** is recommended for deployment due to its highest cross-validation F1-Score (**0.9842**), near-perfect discrimination (**0.9967 ROC-AUC**), fast inference, and clinical interpretability.
