# Heart Failure Survival Analysis

An end-to-end machine learning project predicting 30-day mortality in heart failure patients, built on the [Chicco & Jurman (2020)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5) clinical dataset. Includes exploratory data analysis, statistical testing, unsupervised and supervised learning, hyperparameter optimization, ensemble methods, feature selection, and an interactive Streamlit prediction app.



## Key Findings

- **Serum creatinine** and **ejection fraction** are the strongest predictors of 30-day mortality — consistent with Chicco & Jurman (2020) and supported across all 4 feature selection methods (Lasso, Elastic Net, MRMR, Random Forest)
- **Random Forest** outperformed all other models (Accuracy: 73.3%, ROC-AUC: 0.760, MCC: 0.401) — boosting models (LightGBM, Gradient Boosting) did not surpass it, likely due to the small dataset size (299 patients)
- **Unsupervised clustering** (K-Means, Hierarchical) achieved ~55% accuracy without labels — supervised learning was necessary for meaningful prediction
- **Optuna (Random Sampler)** outperformed GridSearchCV on test accuracy (75.6% vs 73.3%), finding continuous-range parameters that a fixed grid cannot



## Streamlit App

An interactive web app allowing real-time survival prediction based on patient clinical inputs.

```bash
streamlit run predictor_visualization.py
```

**Features:**
- Adjustable sliders for all 11 clinical features
- Real-time survival vs death probability bar chart
- Feature importance visualization
- Risk level classification (Low / Moderate / High)



## Notebooks (Recommended Order)

| # | Notebook | Topics |
|---|---|---|
| 1 | EDA | Distributions, correlations, class imbalance |
| 2 | Statistical Analysis | Hypothesis testing, t-tests, BH-FDR correction |
| 3 | Data Normalization | Z-score, PCA, K-Means, Hierarchical clustering |
| 4 | Supervised Learning & ML | 8 classifiers, confusion matrices, ROC curves |
| 5 | Optimize Machine Learning | GridSearchCV, Optuna Random & Bayesian search |
| 6 | Ensemble Methods | RF vs GB vs LightGBM, MCC metric, Optuna tuning |
| 7 | Feature Selection | Lasso, Elastic Net, MRMR, RF importance, clinical insights |



## Model Comparison Summary

| Model | Accuracy | F1 | ROC-AUC | MCC |
|---|---|---|---|---|
| **Random Forest** | **0.733** | **0.600** | **0.760** | **0.401** |
| Gradient Boosting | 0.733 | 0.586 | 0.714 | 0.390 |
| LightGBM (Tuned) | 0.711 | 0.500 | 0.752 | 0.305 |
| SVM (RBF) | 0.733 | 0.520 | — | — |
| Logistic Regression | 0.711 | 0.458 | — | — |
| KNN | 0.667 | 0.348 | — | — |
| Dummy Baseline | 0.678 | 0.000 | — | — |



## Installation

```bash
git clone https://github.com/rsylee/heart-failure-survival-analysis.git
cd heart-failure-survival-analysis
pip install -r requirements.txt
```



## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
optuna
streamlit
mrmr-selection
scipy
xgboost
jupyter
```



## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Rachel Sooyeon Lee



## Resources

- **Original Paper:** Chicco & Jurman (2020) — BMC Medical Informatics and Decision Making
- **Dataset:** [Kaggle — Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)