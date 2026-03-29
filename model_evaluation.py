"""
Module: Modeling Utilities

This module contains:
- Data preprocessing
- Encoding
- Model evaluation
- Feature importance
"""


# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix
)



# PREPROCESSING


def fill_unknown_with_mode(train_df, val_df, test_df=None, cat_cols=[]):
    """
    Replaces 'unknown' values using mode from training data.
    """
    for col in cat_cols:
        mode_col = train_df[col].mode()[0]
        train_df[col] = train_df[col].replace('unknown', mode_col)
        val_df[col] = val_df[col].replace('unknown', mode_col)

        if test_df is not None:
            test_df[col] = test_df[col].replace('unknown', mode_col)

    return (train_df, val_df, test_df) if test_df is not None else (train_df, val_df)


def encode_columns_safe(train_df, val_df, test_df=None, columns=[], code_map={}):
    """
    Safe categorical encoding using mapping dictionary.
    """
    for col in columns:
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)

        new_col = col + "_code"
        train_df[new_col] = train_df[col].map(code_map)
        val_df[new_col] = val_df[col].map(code_map)

        if test_df is not None:
            test_df[col] = test_df[col].astype(str)
            test_df[new_col] = test_df[col].map(code_map)

    return (train_df, val_df, test_df) if test_df is not None else (train_df, val_df)



# MODEL EVALUATION


def classify_analysis(targets, inputs, model, name=''):
    """
    Evaluates classification model performance.

    Includes:
    - F1 score
    - ROC AUC
    - Confusion matrix
    - ROC curve
    """

    pred_proba = model.predict_proba(inputs)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print(f'F1 score on {name}: {round(f1_score(targets, pred_class), 4)}')

    fpr, tpr, _ = roc_curve(targets, pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f'AUC on {name}: {round(roc_auc, 4)}')
    print(classification_report(targets, pred_class))

    cf = confusion_matrix(targets, pred_class, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cf, annot=True, ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle='--')
    axes[1].set_title("ROC Curve")

    plt.show()



# FEATURE IMPORTANCE


def importance_features(X_train, model, n=10):
    """
    Displays feature importance for tree-based models.
    """
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values('importance', ascending=False)

    display(importance_df.head(n))
    sns.barplot(data=importance_df.head(n), x='importance', y='feature')


def plot_feature_importance(model, train_inputs, top_n=20, normalize=False):
    """
    Displays feature importance for linear models using coefficients.

    Parameters:
    ----------
    model : fitted model
    train_inputs : pd.DataFrame
    top_n : int
    normalize : bool
    """

    coefs = model.coef_[0]
    importance = np.abs(coefs)

    if normalize:
        importance = importance / importance.max()

    fi = pd.Series(importance, index=train_inputs.columns)\
        .sort_values(ascending=False).head(top_n)

    display(fi)

    fi[::-1].plot(kind='barh')
    plt.show()

def get_distribution(df, column, name):
    """
    Returns normalized distribution of a column.
    """
    return df[column].value_counts(normalize=True).rename(name)