# Hyperparameter Tuning and Optimization

Internship task optimizing classification models on the Breast Cancer Wisconsin Diagnostic Dataset using `GridSearchCV` and `RandomizedSearchCV`.

## Repository Structure
- `data/breast_cancer_cleaned.csv`: IQR outlier-capped dataset.
- `src/hyperparameter_tuning.py`: Script running baseline vs. tuned comparisons via `GridSearchCV` and `RandomizedSearchCV`.
- `reports/hyperparameter_tuning_report.md`: Complete report summarizing search spaces, results, and hyperparameter impact.
- `reports/tuning_comparison_results.csv`: Tabular before-and-after metrics.
- `reports/before_after_tuning_chart.png`: Bar chart comparing baseline vs. tuned test F1-Scores.
- `reports/hyperparameter_sensitivity_curve.png`: Regularization sensitivity curve across `C` and `L1/L2` penalties.

## How to Run
```bash
pip install -r requirements.txt
python src/hyperparameter_tuning.py
```
