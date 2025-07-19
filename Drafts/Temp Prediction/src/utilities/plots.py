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
def plot_feature_trends_over_time(data, data_cols, 
                                  station_name = None, 
                                  start_time   = None, 
                                  end_time     = None, 
                                  freq         = None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Nếu là dict nhiều trạm
    if isinstance(data, dict):
        station = data
        for name, df in station.items():
            print(f"🔸 Trạm: {name}")

            # Lọc theo khoảng thời gian nếu có
            start_time = pd.to_datetime(start_time)
            end_time   = pd.to_datetime(end_time)
            df_filtered = df.copy()
            if start_time:
                df_filtered = df_filtered[df_filtered['time'] >= start_time]
            if end_time:
                df_filtered = df_filtered[df_filtered['time'] <= end_time]

            # Resample theo freq nếu có (giả sử cột time đã là datetime và set index)
            if freq:
                df_filtered['time'] = pd.to_datetime(df_filtered['time'])
                df_filtered = df_filtered.set_index('time')

                # Chỉ giữ các cột số
                numeric_cols = df_filtered.select_dtypes(include='number').columns
                df_filtered = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()


            # Duyệt từng cặp feature
            for i in range(0, len(data_cols), 2):
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))

                feature1 = data_cols[i]
                sns.lineplot(data = df_filtered, 
                             x    = "time", 
                             y    = feature1,
                             ax   = axes[0])
                axes[0].set_title(f"Lineplot of {feature1} - {name}")
                axes[0].set_xlabel('Time')
                axes[0].set_ylabel('Value')
                axes[0].grid(True)

                sns.histplot(data = df_filtered, 
                             x    = feature1, 
                            #  y    = feature1,
                             kde  = True, 
                             ax   = axes[1])
                axes[1].set_title(f"Histplot of {feature1} - {name}")
                axes[1].set_xlabel(name)
                axes[1].set_ylabel('Count')
                axes[1].grid(True)

                plt.tight_layout()
                plt.show()

    # Nếu là 1 DataFrame
    elif isinstance(data, pd.DataFrame):
        name = station_name if station_name is not None else "Unknown"
        print(f"🔸 Trạm: {name}")

        # Lọc theo khoảng thời gian nếu có
        start_time = pd.to_datetime(start_time)
        end_time   = pd.to_datetime(end_time)
        df_filtered = data.copy()
        if start_time:
            df_filtered = df_filtered[df_filtered['time'] >= start_time]
        if end_time:
            df_filtered = df_filtered[df_filtered['time'] <= end_time]

        if freq:
            df_filtered['time'] = pd.to_datetime(df_filtered['time'])
            df_filtered = df_filtered.set_index('time')
            
            # Chỉ giữ các cột số
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()


        for i in range(0, len(data_cols)):
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            feature1 = data_cols[i]
            sns.lineplot(data = df_filtered, 
                         x    = "time", 
                         y    = feature1, 
                         ax   = axes[0])
            axes[0].set_title(f"Lineplot of {feature1} - {name}")
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Value')
            axes[0].grid(True)

            sns.histplot(data = df_filtered,
                         x    = feature1, 
                        # y    = feature1,
                         kde  = True,
                         ax   = axes[1])
            axes[1].set_title(f"Histplot of {feature1} - {name}")
            axes[1].set_xlabel(name)
            axes[1].set_ylabel('Count')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    else:
        raise ValueError("Tham số 'data' phải là dict các DataFrame hoặc 1 DataFrame.")
def plot_feature_outliers_over_time(data, data_cols,
                                    station_name      = None,
                                    method            = "statistic",
                                    start_time        = None,
                                    end_time          = None,
                                    freq              = None,
                                    z_thresh          = 3,
                                    modified_z_thresh = 3.5, 
                                    outliers_fraction = 0.2,
                                    n_neighbors       = 20,
                                    factor            = 1.5):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import median_abs_deviation
    import numpy as np

    if isinstance(data, pd.DataFrame):
        name = station_name if station_name else "Unknown"
        print(f"🔸 Trạm: {name}")

        start_time_ = pd.to_datetime(start_time) if start_time else None
        end_time_   = pd.to_datetime(end_time)   if end_time else None
        df_filtered = data.copy()
        df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['time'])

        if start_time_ is not None:
            df_filtered = df_filtered[df_filtered['time'] >= start_time_]
        if end_time_ is not None:
            df_filtered = df_filtered[df_filtered['time'] <= end_time_]

        if freq:
            df_filtered = df_filtered.set_index('time')
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()

        df_filtered = df_filtered.set_index('time')

        for feature in data_cols:
            if method == "statistic":
                fig, axes = plt.subplots(2, 2, figsize=(20, 10))
                
                for row, sub_method in enumerate(["z_score", "z_score modified"]):
                    if sub_method == "z_score":
                        mean = df_filtered[feature].mean()
                        std  = df_filtered[feature].std()
                        bound = (df_filtered[feature] - mean) / std
                        outliers = df_filtered[abs(bound) > z_thresh][feature]

                        print(f"🔹 {feature} ({sub_method}, z_thresh={z_thresh}): {len(outliers)} outliers ~ {len(outliers)/len(df_filtered[feature]):.2%}")

                    elif sub_method == "z_score modified":
                        median = df_filtered[feature].median()
                        MAD    = median_abs_deviation(df_filtered[feature], nan_policy='omit')
                        if MAD == 0:
                            bound = pd.Series([0] * len(df_filtered), index=df_filtered.index)
                        else:
                            bound = 0.6745 * (df_filtered[feature] - median) / MAD
                        outliers = df_filtered[abs(bound) > modified_z_thresh][feature]

                        print(f"🔹 {feature} ({sub_method}, modified_z_thresh={modified_z_thresh}): {len(outliers)} outliers ~ {len(outliers)/len(df_filtered[feature]):.2%}")

                    # Vẽ lineplot
                    axes[row, 0].plot(df_filtered.index, df_filtered[feature],
                                         color     = 'dimgray', 
                                         linestyle = '-', 
                                         alpha     = 0.7, 
                                         label     = f'{feature} (Full Series)')
                    if not outliers.empty:
                        axes[row, 0].scatter(outliers.index, outliers,
                                             color  = 'red', 
                                             label  = 'Outliers', 
                                             marker = 'o')

                    axes[row, 0].set_title(f'{feature} - {sub_method} - {name}')
                    axes[row, 0].set_xlabel('Time')
                    axes[row, 0].set_ylabel('Value')
                    axes[row, 0].grid(True)
                    axes[row, 0].legend()

                    # Barplot số outlier theo năm
                    if not outliers.empty:
                        outlier_counts = outliers.resample('Y').count().astype(int)
                        outlier_counts = outlier_counts.reset_index()
                        outlier_counts['Year'] = outlier_counts['time'].dt.year

                        colors = plt.cm.viridis(np.linspace(0, 1, len(outlier_counts)))
                        axes[row, 1].bar(outlier_counts['Year'], outlier_counts[feature], color=colors)
                        axes[row, 1].set_xticks(outlier_counts['Year'])
                        axes[row, 1].set_ylabel('Number of Outliers')
                        axes[row, 1].set_title(f'Outlier Count per Year - {sub_method}')
                        axes[row, 1].set_xlabel('Year')
                        axes[row, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
                    else:
                        axes[row, 1].text(0.5, 0.5, 'No outliers found.',
                                          horizontalalignment = 'center',
                                          verticalalignment   = 'center',
                                          transform           = axes[row, 1].transAxes,
                                          fontsize            = 14)
                        axes[row, 1].set_title(f'Outlier Count per Year - {sub_method}')
                        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

                    axes[row, 1].tick_params(axis='x', rotation=45)

    else:
        raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")

    
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