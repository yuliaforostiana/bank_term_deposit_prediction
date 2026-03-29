# Bank Term Deposit Prediction

## Problem Description

The task is a **binary classification** problem from the banking domain.

**The goal** is to build a model that predicts whether a client will subscribe to a **term deposit** following a direct marketing phone campaign.

The dataset used is the well-known [Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (`bank-additional-full.csv`), containing **41,188 observations** and **21 features** (10 numerical, 11 categorical). The target variable `y` indicates whether the client subscribed to a term deposit (`yes` / `no`).

A key challenge is the **strong class imbalance**: only ~11.3% of clients subscribed (positive class), while ~88.7% did not.

---

## What Was Done

### 1. Exploratory Data Analysis (EDA)

- Structural checks: data types, missing values, duplicates (12 duplicate pairs found and removed)
- Univariate analysis of all numerical and categorical features
- Class imbalance analysis of the target variable
- Correlation analysis of numerical features (separate matrices for `y=0` and `y=1`)
- Bivariate analysis: numeric vs. numeric, categorical vs. categorical, numeric vs. categorical
- Key finding: macroeconomic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`) and call duration showed the strongest associations with the target

### 2. Data Preprocessing

- **Deduplication**: removed 12 duplicate rows
- **Train/Validation/Test split**: time-based split using `TimeSeriesSplit(n_splits=3)` — months before October for train/validation, November–December for test
- **Missing value imputation**: `unknown` values filled with the mode (fit on train only)
- **Encoding**:
  - Binary encoding for `default`, `housing`, `loan`, `contact`
  - Custom mapping for `poutcome` (`nonexistent=0`, `failure=-1`, `success=1`)
  - Ordinal encoding for `day_of_week`, `education`
  - One-Hot encoding for `job`, `marital`
- **Feature scaling**: `StandardScaler` (fit on train only)
- **Class balancing strategies**: SMOTE, ADASYN, SMOTE-Tomek

### 3. Modelling — Three Groups of Experiments

**Group 1 — Original features, original preprocessing**

Models trained on raw encoded features with different balancing strategies and hyperparameter tuning:
- Logistic Regression (default and GridSearchCV)
- k-Nearest Neighbors (default and GridSearchCV)
- Decision Tree (default and RandomizedSearchCV)
- Random Forest (default and RandomizedSearchCV)
- XGBoost (default, RandomizedSearchCV and Hyperopt)
- LightGBM (default, RandomizedSearchCV and Hyperopt)

**Group 2 — Polynomial features (degree=2)**

Expanded feature space with polynomial interactions between numerical features, applied to Logistic Regression (GridSearchCV + SMOTE) and LightGBM (Hyperopt + ADASYN).

**Group 3 — Manual feature engineering**

Handcrafted interaction features selected based on polynomial feature importance results:
- `macro_stress = emp.var.rate / cons.conf.idx`
- `finance_balance = euribor3m − cons.price.idx`
- `confidence_adjusted = cons.conf.idx × emp.var.rate`
- `contact_intensity = campaign / (previous + 1)`
- `was_contacted_before`, `is_new_client` (binary flags)
- Pairwise interactions: `age × euribor3m`, `age × campaign`, `campaign × emp.var.rate`, etc.

Applied to Logistic Regression (GridSearchCV,  SMOTE) and LightGBM (Hyperopt, ADASYN).

### 4. Best Model Analysis

- SHAP values computed for the best model (LightGBM baseline, Group 1)
- Top drivers: `emp.var.rate` (negative values push toward deposit), `contact_code` (cellular strongly positive), `euribor3m` (low rates → higher subscription probability)
- Error analysis on test set revealed a distribution shift: the test set covers November–December, while training covered March–October

---

## Models Used

| Category | Models |
|---|---|
| Linear | Logistic Regression |
| Distance-based | k-Nearest Neighbors (kNN) |
| Tree-based | Decision Tree |
| Ensemble | Random Forest, XGBoost, LightGBM |
| Balancing methods | SMOTE, ADASYN, SMOTE-Tomek, `scale_pos_weight` |
| Hyperparameter tuning | GridSearchCV, RandomizedSearchCV, Hyperopt |
| Feature engineering | Polynomial features (degree=2), manual interaction features |

---

## Experiment Results
 
The primary metric is **F1-score** on the validation set (positive class). Additional metrics: Recall, Precision, AUROC.
 
| # | Model | Balancing | Tuning | Key Hyperparameters | F1 train | Recall train | Precision train | AUROC train | F1 val | Recall val | Precision val | AUROC val |
|---|-------|-----------|--------|---------------------|----------|--------------|-----------------|-------------|--------|------------|---------------|-----------|
| 1 | Logistic Regression | — | default | `solver=liblinear` | 0.281 | 0.178 | 0.672 | 0.776 | 0.331 | 0.210 | 0.790 | 0.787 |
| 2 | Logistic Regression | SMOTE | default | `solver=liblinear` | 0.705 | 0.663 | 0.751 | 0.794 | 0.503 | 0.630 | 0.420 | 0.786 |
| 3 | Logistic Regression | ADASYN | default | `solver=liblinear` | 0.666 | 0.635 | 0.702 | 0.749 | 0.482 | 0.610 | 0.400 | 0.775 |
| 4 | Logistic Regression | SMOTE-TOMEK | default | `solver=liblinear` | 0.706 | 0.666 | 0.753 | 0.796 | 0.502 | 0.630 | 0.420 | 0.786 |
| 5 | Logistic Regression | SMOTE | GridSearchCV | `C=0.01, penalty=l1, solver=liblinear` | 0.706 | 0.675 | 0.740 | 0.793 | 0.512 | 0.640 | 0.430 | 0.787 |
| 6 | kNN | — | default | `n_neighbors=5` | 0.463 | 0.342 | 0.719 | 0.927 | 0.425 | 0.354 | 0.540 | 0.743 |
| 7 | kNN | SMOTE | default | `n_neighbors=5` | 0.910 | 0.973 | 0.854 | 0.981 | 0.387 | 0.590 | 0.290 | 0.717 |
| 8 | kNN | ADASYN | default | `n_neighbors=5` | 0.903 | 0.976 | 0.840 | 0.980 | 0.381 | 0.580 | 0.280 | 0.706 |
| 9 | kNN | SMOTE-TOMEK | default | `n_neighbors=5` | 0.911 | 0.974 | 0.856 | 0.982 | 0.389 | 0.590 | 0.290 | 0.717 |
| 10 | kNN | — | GridSearchCV | `n_neighbors=15, weights=uniform, p=2` | 0.359 | 0.245 | 0.672 | 0.867 | 0.450 | 0.310 | 0.830 | 0.775 |
| 11 | kNN | SMOTE | GridSearchCV | `n_neighbors=3, weights=distance, p=2` | 0.996 | 0.998 | 0.994 | 1.000 | 0.381 | 0.580 | 0.280 | 0.695 |
| 12 | Decision Tree | — | default | `max_depth=None` | 0.971 | 0.992 | 0.951 | 1.000 | 0.375 | 0.400 | 0.350 | 0.645 |
| 13 | Decision Tree | SMOTE | default | `max_depth=None` | 0.997 | 0.999 | 0.994 | 1.000 | 0.324 | 0.370 | 0.290 | 0.612 |
| 14 | Decision Tree | ADASYN | default | `max_depth=None` | 0.997 | 0.999 | 0.994 | 1.000 | 0.280 | 0.310 | 0.260 | 0.579 |
| 15 | Decision Tree | SMOTE-TOMEK | default | `max_depth=None` | 0.997 | 0.999 | 0.994 | 1.000 | 0.324 | 0.370 | 0.290 | 0.610 |
| 16 | Decision Tree | — | RandomizedSearchCV | `max_depth=5, criterion=entropy, min_samples_leaf=10, min_samples_split=2, max_features=log2` | 0.268 | 0.166 | 0.692 | 0.767 | 0.402 | 0.490 | 0.340 | 0.793 |
| 17 | Decision Tree | SMOTE | RandomizedSearchCV | `max_depth=None, criterion=entropy, min_samples_leaf=2, min_samples_split=10, max_features=None` | 0.963 | 0.963 | 0.963 | 0.996 | 0.358 | 0.420 | 0.310 | 0.656 |
| 18 | Random Forest | — | default | `n_estimators=100` | 0.970 | 0.954 | 0.986 | 1.000 | 0.429 | 0.370 | 0.510 | 0.767 |
| 19 | Random Forest | SMOTE | default | `n_estimators=100` | 0.997 | 0.997 | 0.997 | 1.000 | 0.452 | 0.560 | 0.380 | 0.759 |
| 20 | Random Forest | ADASYN | default | `n_estimators=100` | 0.997 | 0.996 | 0.997 | 1.000 | 0.442 | 0.540 | 0.370 | 0.751 |
| 21 | Random Forest | SMOTE-TOMEK | default | `n_estimators=100` | 0.997 | 0.996 | 0.997 | 1.000 | 0.445 | 0.550 | 0.370 | 0.745 |
| 22 | Random Forest | SMOTE | RandomizedSearchCV | `n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features=log2, bootstrap=False` | 0.982 | 0.973 | 0.992 | 0.999 | 0.468 | 0.440 | 0.500 | 0.761 |
| 23 | XGBoost | — (scale_pos_weight) | default | `max_depth=35, n_estimators=80, scale_pos_weight=8.85` | 0.966 | 1.000 | 0.935 | 1.000 | 0.420 | 0.390 | 0.460 | 0.758 |
| 24 | XGBoost | SMOTE | default | `max_depth=35, n_estimators=80` | 0.996 | 1.000 | 0.992 | 1.000 | 0.440 | 0.540 | 0.370 | 0.749 |
| 25 | XGBoost | ADASYN | default | `max_depth=35, n_estimators=80` | 0.996 | 1.000 | 0.992 | 1.000 | 0.424 | 0.447 | 0.426 | 0.750 |
| 26 | XGBoost | SMOTE-TOMEK | default | `max_depth=35, n_estimators=80` | 0.996 | 1.000 | 0.992 | 1.000 | 0.436 | 0.447 | 0.426 | 0.750 |
| 27 | XGBoost | SMOTE | RandomizedSearchCV | `n_estimators=100, max_depth=35, lr=0.05, subsample=0.8, colsample_bytree=0.6, gamma=0.3, min_child_weight=1` | 0.976 | 1.000 | 0.953 | 1.000 | 0.473 | 0.620 | 0.380 | 0.773 |
| 28 | XGBoost | SMOTE | Hyperopt (20 iter.) | `n_estimators=100, max_depth=15, lr=0.121, subsample=0.713, colsample_bytree=0.937, reg_alpha=0.722, reg_lambda=0.990` | 0.954 | 0.999 | 0.913 | 0.998 | 0.471 | 0.630 | 0.370 | 0.770 |
| **29** | **LightGBM** | **— (scale_pos_weight)** | **default** | **`max_depth=35, n_estimators=80, scale_pos_weight=8.85`** | **0.481** | **0.694** | **0.369** | **0.874** | **0.528** | **0.710** | **0.420** | **0.793** |
| 30 | LightGBM | SMOTE | default | `max_depth=35, n_estimators=80` | 0.841 | 0.981 | 0.736 | 0.974 | 0.347 | 0.832 | 0.219 | 0.792 |
| 31 | LightGBM | SMOTE-TOMEK | default | `max_depth=35, n_estimators=80` | 0.843 | 0.983 | 0.738 | 0.975 | 0.359 | 0.824 | 0.229 | 0.795 |
| 32 | LightGBM | — | RandomizedSearchCV | `n_estimators=100, max_depth=5, lr=0.01, num_leaves=15, subsample=0.8, colsample_bytree=1.0` | 0.466 | 0.473 | 0.460 | 0.803 | 0.523 | 0.690 | 0.420 | 0.793 |
| 33 | LightGBM | ADASYN | RandomizedSearchCV | `n_estimators=300, max_depth=5, lr=0.05, num_leaves=50, subsample=0.8, colsample_bytree=0.6, reg_lambda=0.5` | 0.888 | 0.963 | 0.824 | 0.974 | 0.427 | 0.540 | 0.350 | 0.784 |
| 34 | LightGBM | — | Hyperopt (20 iter.) | `n_estimators=235, max_depth=13, lr=0.111, num_leaves=29, subsample=0.823, colsample_bytree=0.510, scale_pos_weight=1` | 0.436 | 0.304 | 0.767 | 0.858 | 0.438 | 0.560 | 0.360 | 0.791 |
| 35 | LightGBM | ADASYN | Hyperopt (20 iter.) | `n_estimators=255, max_depth=5, lr=0.019, num_leaves=31, subsample=0.945, colsample_bytree=0.787, scale_pos_weight=3` | 0.832 | 0.956 | 0.737 | 0.945 | 0.378 | 0.500 | 0.300 | 0.778 |
| 36 | Logistic Regression + poly² | SMOTE | GridSearchCV | `C=0.01, penalty=l2, solver=liblinear` | 0.709 | 0.677 | 0.743 | 0.783 | 0.513 | 0.640 | 0.430 | 0.787 |
| 37 | LightGBM + poly² | ADASYN | Hyperopt (20 iter.) | `max_depth=5, lr=0.019, num_leaves=31, subsample=0.945` | 0.879 | 0.853 | 0.907 | 0.940 | 0.435 | 0.540 | 0.370 | 0.739 |
| 38 | Logistic Regression + new features | SMOTE | GridSearchCV | `C=0.01, penalty=l2, solver=liblinear` | 0.742 | 0.705 | 0.784 | 0.833 | 0.496 | 0.620 | 0.410 | 0.765 |
| 39 | LightGBM + new features | ADASYN | Hyperopt (20 iter.) | `max_depth=5, lr=0.019, num_leaves=31, subsample=0.944` | 0.971 | 0.962 | 0.979 | 0.997 | 0.480 | 0.500 | 0.460 | 0.774 |
 
---

## Conclusions

### What Was Achieved

**Best model: LightGBM (Group 1, unbalanced data with `scale_pos_weight`)** — experiment #29

- **F1 = 0.528**, Recall = 0.71, Precision = 0.42, AUROC = 0.793 on validation
- Achieved the highest F1 and recall across all 39 experiments without synthetic oversampling or hyperparameter tuning
- SHAP analysis confirmed interpretable and business-logical feature importance: macroeconomic conditions (`emp.var.rate`, `euribor3m`) and contact channel (`contact_code`) were the dominant drivers

**Key findings across experiments:**

- Synthetic oversampling (SMOTE/ADASYN/SMOTE-Tomek) consistently improved recall for most models but hurt precision and AUROC, particularly for LightGBM where it degraded F1 significantly
- `scale_pos_weight` was a more effective strategy for gradient boosting models than synthetic sampling
- Polynomial features and manual feature engineering did not improve over the LightGBM baseline; the interactions were already captured by the tree structure
- Decision Tree was the weakest model overall, especially prone to overfitting on default settings (AUROC 0.645 on validation vs. near-perfect on training)
- kNN showed the largest train/validation gap, with the tuned SMOTE variant achieving F1=0.38 at train F1=0.996

**Worst model: kNN (tuned, SMOTE and GridSearchCV, k=3)** — experiment #11

- F1 = 0.381 on validation, Precision = 0.28 (only 1 in 3 positive predictions was correct)
- Extreme overfitting: train AUROC = 0.9999 vs. validation AUROC = 0.695

**Test set issue:** The best model dropped from F1=0.528 (validation) to F1=0.241 (test). AUROC remained stable at 0.79, indicating the model retains discriminative ability but the default 0.5 threshold becomes suboptimal on the November–December test months due to a distribution shift in both client behaviour and macroeconomic indicators.

---

## What Could Be Improved

1. **Threshold optimisation** — tune the classification threshold specifically for the test distribution rather than using the default 0.5
2. **Distribution shift analysis** — investigate why November–December behave differently; consider calendar features or temporal reweighting
3. **More Hyperopt iterations** — 20 iterations is insufficient for the large hyperparameter spaces used; 100+ iterations could yield materially better results
4. **Feature selection** — remove low-importance features to reduce noise and improve generalisation
5. **Ensemble of strong learners** — combine Logistic Regression (good calibration) and LightGBM (strong discrimination) via stacking or blending
6. **Multi-year data** — the dataset covers a single year; training on multiple years would allow the model to learn seasonal patterns properly
7. **Alternative metrics / cost-sensitive learning** — in a real banking context, the cost of missing a potential subscriber (false negative) differs from the cost of unnecessary outreach (false positive); optimising for business-weighted metrics could further improve deployment value

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `XGBoost` · `LightGBM` · `Hyperopt` · `SHAP` · `matplotlib` · `seaborn` · `plotly`
