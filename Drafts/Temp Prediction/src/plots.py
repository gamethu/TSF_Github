# Mutual infomation
def make_mi_scores(X_data, y_data):
    X_data = X_data.copy()
    for colname in X_data.select_dtypes(["object", "category"]):
        X_data[colname], _ = X_data[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X_data.dtypes]
    mi_scores = mutual_info_classif(X_data, y_data, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores):
    plt.grid(True, axis='x')
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")