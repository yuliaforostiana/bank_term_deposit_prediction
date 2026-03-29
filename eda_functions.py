"""
Module: EDA Utilities

This module contains functions for:
- Exploratory Data Analysis (EDA)
- Visualization
- Outlier detection
- Feature relationship analysis
"""


# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



# BASIC EDA


def exploration_analysis(df, df_column):
    """
    Performs univariate analysis for a column.

    Includes:
    - Missing values
    - Value counts
    - Numeric distributions (hist, QQ, boxplot)
    - Descriptive statistics
    - Outlier thresholds (IQR)

    Parameters:
    ----------
    df : pd.DataFrame
    df_column : str
    """
    print("Missing values count:", df[df_column].isnull().sum())
    print("Missing values percent of total, %:", round((df[df_column].isnull().sum() / df.shape[0]) * 100, 2))

    print("Column values")
    counts = df[df_column].value_counts()
    share = ((counts / len(df)) * 100).round(2)
    null_column_values = pd.DataFrame({'Count': counts, 'Proportion': share})
    display(null_column_values.sort_values(by='Proportion', ascending=False).head(15))

    if pd.api.types.is_numeric_dtype(df[df_column]):
        fig, axs = plt.subplots(1, 3, figsize=(16, 7))

        sns.histplot(df[df_column], bins=25, ax=axs[0])
        stats.probplot(df[df_column], dist="norm", plot=axs[1])
        sns.boxplot(df[df_column], ax=axs[2])

        plt.show()

        display(df[df_column].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99, 0.999]).round(4))

    else:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=df[df_column])
        plt.xticks(rotation=90)
        plt.show()



# CATEGORICAL ANALYSIS


def bi_cat_countplot(df, column, hue_column):
    """
    Plots categorical distribution with normalization and counts.

    Parameters:
    ----------
    df : pd.DataFrame
    column : str
    hue_column : str
    """
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    proportions = df.groupby(hue_column)[column].value_counts(normalize=True) * 100
    proportions.unstack(hue_column).plot.bar(ax=axes[0])

    counts = df.groupby(hue_column)[column].value_counts()
    counts.unstack(hue_column).plot.bar(ax=axes[1])


def uni_cat_y_compare(df, column):
    """
    Wrapper to compare categorical feature vs target 'y'.
    """
    bi_cat_countplot(df, column, 'y')


def bi_countplot_y(df0, df1, column, hue_column):
    """
    Compares categorical distributions for two datasets.

    Parameters:
    ----------
    df0 : pd.DataFrame
    df1 : pd.DataFrame
    column : str
    hue_column : str
    """
    print("TARGET = 1")
    bi_cat_countplot(df1, column, hue_column)
    plt.show()

    print("TARGET = 0")
    bi_cat_countplot(df0, column, hue_column)
    plt.show()


def bi_countplot_target(df0, df1, column, hue_column):
    """
    Extended categorical comparison (counts + proportions) per target group.
    """

    group_name = f'Нормалізований розподіл значень за категорією: {column}'
    print(group_name.upper())

    unique_hue_values = df1[hue_column].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    proportions = df1.groupby(hue_column)[column].value_counts(normalize=True) * 100
    proportions.unstack(hue_column).plot.bar(ax=axes[0])

    proportions = df0.groupby(hue_column)[column].value_counts(normalize=True) * 100
    proportions.unstack(hue_column).plot.bar(ax=axes[1])

    plt.show()



# OUTLIERS & NUMERIC


def kde_no_outliers(df0, df1, Max_value0, Max_value1, column):
    """
    Plots KDE distributions after removing outliers.

    Parameters:
    ----------
    df0 : pd.DataFrame
    df1 : pd.DataFrame
    Max_value0 : float
    Max_value1 : float
    column : str
    """
    plt.figure(figsize=(14,6))
    sns.kdeplot(df1[df1[column] <= Max_value1][column])
    sns.kdeplot(df0[df0[column] <= Max_value0][column])
    plt.show()


def one_dimentional_numeric(df, column, target):
    """
    Performs numeric analysis per target class with outlier detection.
    """
    df1 = df[df[target] == 1]
    df0 = df[df[target] == 0]

    Max_value1 = outlier_range(df1, column)
    Max_value0 = outlier_range(df0, column)

    kde_no_outliers(df0, df1, Max_value0, Max_value1, column)


def outlier_range(dataset, column):
    """
    Calculates IQR-based upper outlier threshold.

    Returns:
    -------
    float
    """
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    return Q3 + 1.5 * (Q3 - Q1)



# MULTIVARIATE ANALYSIS


def multidimensional_numeric(df, columns):
    """
    Scatterplot comparison for two numeric features by target class.
    """
    df1 = df[df['y'] == 'yes']
    df0 = df[df['y'] == 'no']

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    sns.scatterplot(x=df1[columns[0]], y=df1[columns[1]])

    plt.subplot(1,2,2)
    sns.scatterplot(x=df0[columns[0]], y=df0[columns[1]])

    plt.show()


def draw_boxplot(df, categorical, continuous, max_continuous, title, hue_column, subplot_position):
    """
    Draws a boxplot for categorical vs numeric feature.
    """
    plt.subplot(1, 2, subplot_position)
    sns.boxplot(
        x=categorical,
        y=df[df[continuous] < max_continuous][continuous],
        data=df,
        hue=hue_column
    )


def bi_boxplot(categorical, continuous, max_continuous1, max_continuous0, hue_column, df):
    """
    Boxplot comparison for two target classes.
    """
    df1 = df[df['y'] == 'yes']
    df0 = df[df['y'] == 'no']

    plt.figure(figsize=(16, 10))
    draw_boxplot(df1, categorical, continuous, max_continuous1, 'Success', hue_column, 1)
    draw_boxplot(df0, categorical, continuous, max_continuous0, 'Failure', hue_column, 2)
    plt.show()


def numeric_vs_categorical_corr(df, columns):
    """
    Analyzes relationship between numeric and categorical features.
    """
    df1 = df[df['y'] == 'yes']
    df0 = df[df['y'] == 'no']

    numeric_columns = df[columns].select_dtypes(include=["number"]).columns
    categorical_columns = df[columns].select_dtypes(exclude=["number"]).columns

    print("TARGET = YES")
    display(df1.groupby(list(categorical_columns))[numeric_columns].describe())

    print("TARGET = NO")
    display(df0.groupby(list(categorical_columns))[numeric_columns].describe())
