from sklearn.neighbors import LocalOutlierFactor
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
                                    display           = False,
                                    start_time        = None,
                                    end_time          = None,
                                    freq              = None,
                                    z_thresh          = 3,
                                    modified_z_thresh = 3.5, 
                                    models            = dict({"LocalOutlierFactor" : LocalOutlierFactor()}),
                                    factor            = 1.5,
                                    window_size       = 10,
                                    dendrogram        = False):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import median_abs_deviation
    from copy import deepcopy
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.abspath("../src"))


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
                
                from models.anomaly_models import (MyZ_Score,
                                                   MyZ_Score_modified)
                
                for row, sub_method in enumerate(["z_score", "z_score modified"]):
                    if sub_method == "z_score":
                        Z_outlier = MyZ_Score(data      = df_filtered,
                                              data_cols = feature,
                                              display   = display,
                                              z_thresh  = z_thresh,
                                              ax        = list([axes[0,0], axes[0,1]]))
                        print(f"🔹 {feature} (Z_Score, z_thresh={z_thresh}): {len(Z_outlier)} outliers ~ {len(Z_outlier)/len(df_filtered[feature]):.2%}")

                    elif sub_method == "z_score modified":
                        ZM_outlier = MyZ_Score_modified(data              = df_filtered,
                                                        data_cols         = feature,
                                                        display           = display,
                                                        modified_z_thresh = modified_z_thresh,
                                                        ax                  = list([axes[1,0], axes[1,1]]))
                        print(f"🔹 {feature} (Z_Score_Modified, modified_z_thresh={modified_z_thresh}): {len(ZM_outlier)} outliers ~ {len(ZM_outlier)/len(df_filtered[feature]):.2%}")
            elif method == "machine_learning":        
                fig, axes = plt.subplots(4, 2, figsize=(20, 20))
                
                from models.anomaly_models import (MyIsolationForest,
                                                   MyLocalOutlierFactor,
                                                   MyProphet,
                                                   MyAgglomerativeClustering,
                                                   MyDBSCAN,
                                                   MyVanillaAutoencoder)
                # Option 1
                if models.get("IsolationForest") is not None:
                    MIF_model   = models.get("IsolationForest")
                    MIF_outlier = MyIsolationForest(data      = df_filtered,
                                                    data_cols = feature,
                                                    model     = MIF_model,
                                                    display   = display,
                                                    ax        = axes[0,0])
                    print(f"🔹 {feature} (IsolationForest, {MIF_model}): {len(MIF_outlier)} outliers ~ {len(MIF_outlier)/len(df_filtered[feature]):.2%}")
                
                # Option 2
                if models.get("LocalOutlierFactor") is not None:
                    MLOF_model   = models.get("LocalOutlierFactor")
                    MLOF_outlier = MyLocalOutlierFactor(data      = df_filtered,
                                                        data_cols = feature,
                                                        model     = MLOF_model,
                                                        display   = display,
                                                        ax        = axes[0,1])
                    print(f"🔹 {feature} (LocalOutlierFactor, {MLOF_model}): {len(MLOF_outlier)} outliers ~ {len(MLOF_outlier)/len(df_filtered[feature]):.2%}")
                
                # Option 3
                if models.get("Prophet") is not None:
                    MP_model   = deepcopy(models.get("Prophet"))
                    MP_outlier = MyProphet(data      = df_filtered.reset_index(), # Slow!!!
                                           data_cols = feature,
                                           model     = MP_model,
                                           display   = display,
                                           factor    = factor,
                                           ax        = list([axes[1,0],axes[1,1]]))
                    print(f"🔹 {feature} (Prophet, {MP_model}): {len(MP_outlier)} outliers ~ {len(MP_outlier)/len(df_filtered[feature]):.2%}")
                
                # Option 4
                if models.get("AgglomerativeClustering") is not None:
                    MAC_model   = deepcopy(models.get("AgglomerativeClustering"))
                    MAC_outlier = MyAgglomerativeClustering(data        = df_filtered.reset_index(), # Slow!!!
                                                            data_cols   = feature,
                                                            model       = MAC_model,
                                                            display     = display,
                                                            window_size = window_size,
                                                            dendrogram  = dendrogram,
                                                            ax          = list([axes[2,0],axes[2,1]]))
                    print(f"🔹 {feature} (AgglomerativeClustering, {MAC_model}): {len(MAC_outlier)} outliers ~ {len(MAC_outlier)/len(df_filtered[feature]):.2%}")
                
                # Option 5
                if models.get("DBSCAN") is not None:
                    M_model   = deepcopy(models.get("DBSCAN"))
                    M_outlier = MyDBSCAN(data        = df_filtered.reset_index(), # Slow!!!
                                         data_cols   = feature,
                                         model       = M_model,
                                         display     = display,
                                         window_size = window_size,
                                         ax          = axes[3,0])
                    print(f"🔹 {feature} (DBSCAN, {M_model}): {len(M_outlier)} outliers ~ {len(M_outlier)/len(df_filtered[feature]):.2%}")
                
                # Option 6
                if models.get("VanillaAutoencoder") is not None:
                    # MVA_model   = deepcopy(models.get("VanillaAutoencoder"))
                    MVA_outlier = MyVanillaAutoencoder(data        = df_filtered.reset_index(), # Slow!!!
                                                       data_cols   = feature,
                                                       display     = display,
                                                    #    model       = MVA_model,
                                                       ax          = axes[3,1])
                    print(f"🔹 {feature} (VanillaAutoencoder): {len(MVA_outlier)} outliers ~ {len(MVA_outlier)/len(df_filtered[feature]):.2%}")
                
            else:
                raise ValueError(f"Giá trị method không hợp lệ: {method}")

            if display is True:
                plt.suptitle(f'Outlier Detection for {feature} - {name}', fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
            else:
                plt.close(fig)

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