## Outlier data
def plot_Outlier(data, data_cols, target=None):
    """
    Hiển thị histogram và boxplot cho từng biến số trong data_cols.
    - Nếu có target: hiển thị theo class
    - Nếu không có: hiển thị phân phối và boxplot thông thường
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_cols = len(data_cols)
    ncols = 2
    nrows = num_cols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    for i, column in enumerate(data_cols):
        # Histplot
        sns.histplot(
            data=data,
            x=column,
            hue=target if target else None,
            kde=True,
            ax=axes[i, 0]
        )
        if target:
            axes[i, 0].set_title(f'Histogram: {column} by {target}')
        else:
            axes[i, 0].set_title(f'Histogram: {column}')
        axes[i, 0].grid(True)

        # Boxplot
        if target:
            sns.boxplot(
                data=data,
                x=target,
                y=column,
                ax=axes[i, 1]
            )
            axes[i, 1].set_title(f'Boxplot: {column} by {target}')
            axes[i, 1].set_xlabel(target)
        else:
            sns.boxplot(
                data=data,
                y=column,
                ax=axes[i, 1]
            )
            axes[i, 1].set_title(f'Boxplot: {column}')
        axes[i, 1].set_ylabel(column)
        axes[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

# Mutual infomation
def make_mi_scores_classification(X_data, y_data):
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif
    
    X_data = X_data.copy()
    for colname in X_data.select_dtypes(["object", "category"]):
        X_data[colname], _ = X_data[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X_data.dtypes]
    mi_scores = mutual_info_classif(X_data, y_data, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def make_mi_scores_regression(X_data, y_data):
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression
    
    X_data = X_data.copy()
    for colname in X_data.select_dtypes(["object", "category"]):
        X_data[colname], _ = X_data[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X_data.dtypes]
    mi_scores = mutual_info_regression(X_data, y_data, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores):
    from matplotlib import pyplot as plt
    import numpy as np
    
    plt.grid(True, axis='x')
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")