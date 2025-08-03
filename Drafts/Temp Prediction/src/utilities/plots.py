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
                                    step_size       = 10,
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
                                                            step_size = step_size,
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
                                         step_size = step_size,
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

def evaluate_feature_outliers_over_time(data, data_cols,
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
                                    step_size       = 10,
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
            df_filtered  = df_filtered.set_index('time')
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered  = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()

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
                fig, axes = plt.subplots(6, 2, figsize=(20, 20))
                
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
                                                    display   = False,
                                                    ax        = None)
                    print(f"🔹 {feature} (IsolationForest, {MIF_model}): {len(MIF_outlier)} outliers ~ {len(MIF_outlier)/len(df_filtered[feature]):.2%}")                    
                    custom_evaluate_model(df_filtered[feature], MIF_outlier, station_name, feature, list([axes[0,0], axes[0,1]]), "IsolationForest")
                # Option 2
                if models.get("LocalOutlierFactor") is not None:
                    MLOF_model   = models.get("LocalOutlierFactor")
                    MLOF_outlier = MyLocalOutlierFactor(data      = df_filtered,
                                                        data_cols = feature,
                                                        model     = MLOF_model,
                                                        display   = False,
                                                        ax        = None)
                    print(f"🔹 {feature} (LocalOutlierFactor, {MLOF_model}): {len(MLOF_outlier)} outliers ~ {len(MLOF_outlier)/len(df_filtered[feature]):.2%}")
                    custom_evaluate_model(df_filtered[feature], MLOF_outlier, station_name, feature, list([axes[1,0], axes[1,1]]), "LocalOutlierFactor")
                # Option 3
                if models.get("Prophet") is not None:
                    MP_model   = deepcopy(models.get("Prophet"))
                    MP_outlier = MyProphet(data      = df_filtered.reset_index(), # Slow!!!
                                           data_cols = feature,
                                           model     = MP_model,
                                           display   = False,
                                           factor    = factor,
                                           ax        = None)
                    print(f"🔹 {feature} (Prophet, {MP_model}): {len(MP_outlier)} outliers ~ {len(MP_outlier)/len(df_filtered[feature]):.2%}")
                    custom_evaluate_model(df_filtered[feature], MP_outlier, station_name, feature, list([axes[2,0], axes[2,1]]), "Prophet")
                # Option 4
                if models.get("AgglomerativeClustering") is not None:
                    MAC_model   = deepcopy(models.get("AgglomerativeClustering"))
                    MAC_outlier = MyAgglomerativeClustering(data        = df_filtered.reset_index(), # Slow!!!
                                                            data_cols   = feature,
                                                            model       = MAC_model,
                                                            display     = False,
                                                            step_size = step_size,
                                                            dendrogram  = dendrogram,
                                                            ax          = None)
                    print(f"🔹 {feature} (AgglomerativeClustering, {MAC_model}): {len(MAC_outlier)} outliers ~ {len(MAC_outlier)/len(df_filtered[feature]):.2%}")
                    custom_evaluate_model(df_filtered[feature], MAC_outlier, station_name, feature, list([axes[3,0], axes[3,1]]), "AgglomerativeClustering")
                # Option 5
                if models.get("DBSCAN") is not None:
                    M_model   = deepcopy(models.get("DBSCAN"))
                    M_outlier = MyDBSCAN(data        = df_filtered.reset_index(), # Slow!!!
                                         data_cols   = feature,
                                         model       = M_model,
                                         display     = False,
                                         step_size = step_size,
                                         ax          = None)
                    print(f"🔹 {feature} (DBSCAN, {M_model}): {len(M_outlier)} outliers ~ {len(M_outlier)/len(df_filtered[feature]):.2%}")
                    custom_evaluate_model(df_filtered[feature], M_outlier, station_name, feature, list([axes[4,0], axes[4,1]]), "DBSCAN")
                # Option 6
                if models.get("VanillaAutoencoder") is not None:
                    # MVA_model   = deepcopy(models.get("VanillaAutoencoder"))
                    MVA_outlier = MyVanillaAutoencoder(data        = df_filtered.reset_index(), # Slow!!!
                                                       data_cols   = feature,
                                                       display     = False,
                                                    #    model       = MVA_model,
                                                       ax          = None)
                    print(f"🔹 {feature} (VanillaAutoencoder): {len(MVA_outlier)} outliers ~ {len(MVA_outlier)/len(df_filtered[feature]):.2%}")
                    custom_evaluate_model(df_filtered[feature], MVA_outlier, station_name, feature, list([axes[5,0], axes[5,1]]), "VanillaAutoencoder")
            else:
                raise ValueError(f"Giá trị method không hợp lệ: {method}")

            # if display is True:
            #     plt.suptitle(f'Outlier Detection for {feature} - {name}', fontsize=18)
            #     plt.tight_layout(rect=[0, 0, 1, 0.96])
            #     plt.show()
            # else:
            #     plt.close(fig)

    else:
        raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")

def custom_evaluate_model(y_true, outlier_idx, station_name, feature_name, ax, model_name):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    n = len(y_true)
    outlier_idx = np.array(outlier_idx)

    # Tính tỷ lệ outlier
    anomaly_ratio = len(outlier_idx) / n if n > 0 else 0

    # 2. Phân bố giá trị outlier vs normal
    normal_values = np.delete(y_true, outlier_idx) if len(outlier_idx) > 0 else y_true

    normal_values = np.delete(y_true, outlier_idx) if len(outlier_idx) > 0 else y_true
    outlier_values = y_true[outlier_idx] if len(outlier_idx) > 0 else np.array([])
    # normal_stats = pd.Series(normal_values).describe().to_dict() if len(normal_values) > 0 else {}
    # outlier_stats = pd.Series(outlier_values).describe().to_dict() if len(outlier_values) > 0 else {}

    mean_deviation = 0
    if len(outlier_values) > 0 and len(normal_values) > 0:
        normal_mean = np.mean(normal_values)
        normal_std = np.std(normal_values)
        outlier_mean = np.mean(outlier_values)
        mean_deviation = abs(outlier_mean - normal_mean) / normal_std if normal_std != 0 else 0

    if ax is not None:
        if len(normal_values) > 0:
            sns.kdeplot(normal_values, label="Normal", color="blue", fill=True, ax=ax[0])
        if len(outlier_idx) > 0:
            sns.kdeplot(y_true[outlier_idx], label="Outliers", color="red", fill=True, ax=ax[0])
        ax[0].set_title(f"{feature_name} (Normal vs Outlier) - {model_name}")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].text(0.02, 0.98, f"Outlier Ratio: {anomaly_ratio:.4f}\nMean Deviation: {mean_deviation:.4f}",
                transform=ax[0].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        df = pd.DataFrame({"Values": y_true})
        sns.boxplot(data=df, y="Values", color="skyblue", showfliers=True, ax=ax[1])  # mặc định boxplot đánh dấu outlier bằng các chấm
        if len(outlier_idx) > 0:
            ax[1].scatter(np.zeros(len(outlier_idx)), y_true[outlier_idx], color="red")
        ax[1].set_title(f"Standard Boxplot with Outliers - {model_name}")
        ax[1].set_ylabel("Value")
        ax[1].grid(True, axis="y")        

# Custom evaluation function to replace plots.evaluate_model
def plot_evaluate_model_over_time(data, target_cols_name, station_name, y_true, y_pred,
                                  method           = "short",
                                  evaluate_metrics = dict({
                                                      "R2"   : "r2_score",   
                                                    #   "MAE"  : "mean_absolute_error",
                                                    #   "MSE"  : "mean_squared_error",
                                                    #   "MSLE" : "mean_squared_log_error",
                                                    #   "MAPE" : "mean_absolute_percentage_error"
                                                      }),
                                  display          = False,
                                  start_time       = None,
                                  end_time         = None,
                                  step_size        = 24,
                                  freq             = None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import median_abs_deviation
    from copy import deepcopy
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.abspath("../../src"))
    
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
            df_filtered  = df_filtered.set_index('time')
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered  = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()

        df_filtered = df_filtered.set_index('time')

        if method == "short":
            from scripts.evaluate_model import (My_R2_SCORE,
                                                My_MAE_SCORE,
                                                My_MSE_SCORE,
                                                My_MSLE_SCORE,
                                                My_MAPE_SCORE)
            # Option 1
            if evaluate_metrics.get("R2") is not None:
                R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                                                            y_pred    = y_pred,
                                                            y_true    = y_true,
                                                            display   = False,
                                                            step_size = step_size,
                                                            freq      = freq,
                                                            ax        = None)
                print(f"🔹 {target_cols_name}_{name} (R2_train) : {R2_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (R2_test)  : {R2_SCORE_TEST}")
                print()                    
            
            # Option 2
            if evaluate_metrics.get("MAE") is not None:
                MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
                                                               y_pred    = y_pred,
                                                               y_true    = y_true,
                                                               display   = False,
                                                               step_size = step_size,
                                                               freq      = freq,
                                                               ax        = None)
                print(f"🔹 {target_cols_name}_{name} (MAE_train) : {MAE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MAE_test)  : {MAE_SCORE_TEST}")
                print()
              
            # Option 3
            if evaluate_metrics.get("MSE") is not None:
                MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
                                                               y_pred    = y_pred,
                                                               y_true    = y_true,
                                                               display   = False,
                                                               step_size = step_size,
                                                               freq      = freq,
                                                               ax        = None)
                print(f"🔹 {target_cols_name}_{name} (MSE_train) : {MSE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MSE_test)  : {MSE_SCORE_TEST}")
                print()
              
            # Option 4
            if evaluate_metrics.get("MSLE") is not None:
                MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
                                                                  y_pred    = y_pred,
                                                                  y_true    = y_true,
                                                                  display   = False,
                                                                  step_size = step_size,
                                                                  freq      = freq,
                                                                  ax        = None)
                print(f"🔹 {target_cols_name}_{name} (MSLE_train) : {MSLE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MSLE_test)  : {MSLE_SCORE_TEST}")
                print()
              
            # Option 5
            if evaluate_metrics.get("MAPE") is not None:
                MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
                                                                  y_pred    = y_pred,
                                                                  y_true    = y_true,
                                                                  display   = False,
                                                                  step_size = step_size,
                                                                  freq      = freq,
                                                                  ax        = None)
                print(f"🔹 {target_cols_name}_{name} (MAPE_train) : {MAPE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MAPE_test)  : {MAPE_SCORE_TEST}")
                print()
              
            # # Option 6
            # if evaluate_metrics.get("R2") is not None:
            #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
            #                                                 y_pred    = y_pred,
            #                                                 y_true    = y_true,
            #                                                 display   = False,
            #                                                 freq      = freq,
            #                                                 ax        = None)
            #     print(f"🔹 {target_cols_name}_{name} (R2_train): {R2_SCORE_TRAIN}")
            #     print(f"🔹 {target_cols_name}_{name} (R2_test): {R2_SCORE_TEST}")
            #     print()
        elif method == "full":        
            fig, axes = plt.subplots(6, 2, figsize=(20, 30))
            
            from scripts.evaluate_model import (My_R2_SCORE,
                                                My_MAE_SCORE,
                                                My_MSE_SCORE,
                                                My_MSLE_SCORE,
                                                My_MAPE_SCORE)
            # Option 1
            if evaluate_metrics.get("R2") is not None:
                R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                                                            y_pred    = y_pred,
                                                            y_true    = y_true,
                                                            display   = display,
                                                            step_size = step_size,
                                                            freq      = freq,
                                                            ax        = list([axes[0,0],axes[0,1]]))
                print(f"🔹 {target_cols_name}_{name} (R2_train) : {R2_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (R2_test)  : {R2_SCORE_TEST}")
                print()                    
            
            # Option 2
            if evaluate_metrics.get("MAE") is not None:
                MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
                                                               y_pred    = y_pred,
                                                               y_true    = y_true,
                                                               display   = display,
                                                               step_size = step_size,
                                                               freq      = freq,
                                                               ax        = list([axes[1,0],axes[1,1]]))
                print(f"🔹 {target_cols_name}_{name} (MAE_train) : {MAE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MAE_test)  : {MAE_SCORE_TEST}")
                print()
              
            # Option 3
            if evaluate_metrics.get("MSE") is not None:
                MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
                                                               y_pred    = y_pred,
                                                               y_true    = y_true,
                                                               display   = display,
                                                               step_size = step_size,
                                                               freq      = freq,
                                                               ax        = list([axes[2,0],axes[2,1]]))
                print(f"🔹 {target_cols_name}_{name} (MSE_train) : {MSE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MSE_test)  : {MSE_SCORE_TEST}")
                print()
              
            # Option 4
            if evaluate_metrics.get("MSLE") is not None:
                MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
                                                                  y_pred    = y_pred,
                                                                  y_true    = y_true,
                                                                  display   = display,
                                                                  step_size = step_size,
                                                                  freq      = freq,
                                                                  ax        = list([axes[3,0],axes[3,1]]))
                print(f"🔹 {target_cols_name}_{name} (MSLE_train) : {MSLE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MSLE_test)  : {MSLE_SCORE_TEST}")
                print()
              
            # Option 5
            if evaluate_metrics.get("MAPE") is not None:
                MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
                                                                  y_pred    = y_pred,
                                                                  y_true    = y_true,
                                                                  display   = display,
                                                                  step_size = step_size,
                                                                  freq      = freq,
                                                                  ax        = list([axes[4,0],axes[4,1]]))
                print(f"🔹 {target_cols_name}_{name} (MAPE_train) : {MAPE_SCORE_TRAIN}")
                print(f"🔹 {target_cols_name}_{name} (MAPE_test)  : {MAPE_SCORE_TEST}")
                print()
              
            # # Option 6
            # if evaluate_metrics.get("R2") is not None:
            #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
            #                                                 y_pred    = y_pred,
            #                                                 y_true    = y_true,
            #                                                 display   = display,
            #                                                 freq      = freq,
            #                                                 ax        = list([axes[0,0],axes[0,1]]))
            #     print(f"🔹 {target_cols_name}_{name} (R2_train): {R2_SCORE_TRAIN}")
            #     print(f"🔹 {target_cols_name}_{name} (R2_test): {R2_SCORE_TEST}")
            #     print()
              
        else:
            raise ValueError(f"Giá trị method không hợp lệ: {method}")

        if display is True:
            plt.suptitle(f'Evaluation Model - {name}', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            plt.close(fig)

    else:
        raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")
def plot_evaluate_params_over_time(data, target_cols_name, station_name, x_fit, y_true, model, params,
                                  method           = "short",
                                  evaluate_metrics = dict({
                                                      "R2"   : "r2_score",   
                                                    #   "MAE"  : "mean_absolute_error",
                                                    #   "MSE"  : "mean_squared_error",
                                                    #   "MSLE" : "mean_squared_log_error",
                                                    #   "MAPE" : "mean_absolute_percentage_error"
                                                      }),
                                  display          = False,
                                  start_time       = None,
                                  end_time         = None,
                                  step_size        = 24,
                                  freq             = None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import median_abs_deviation
    from copy import deepcopy
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.abspath("../../src"))
    
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
            df_filtered  = df_filtered.set_index('time')
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered  = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()

        df_filtered = df_filtered.set_index('time')

        # Set defaul params for model
        model.set_params(**dict({k : v[0] for k,v in params.items()}))
        print("default_model: ", model)
        
        if method == "short":
            from scripts.evaluate_model import (My_R2_SCORE,
                                                My_MAE_SCORE,
                                                My_MSE_SCORE,
                                                My_MSLE_SCORE,
                                                My_MAPE_SCORE)
            global_d     = dict({})
            global_total = len(params)
            global_best  = 0
            for i, key in enumerate(params.keys(), 1):
                print(f"Lap: {i}/{global_total}")
                # Option 1
                if evaluate_metrics.get("R2") is not None:
                    local_d = dict({})
                    for values in params[key]:
                        local_model = deepcopy(model)
                        local_model = local_model.set_params(**{key: values})
                        local_model = local_model.fit(x_fit[0],y_true[0])
                        y_fit       = list([pd.DataFrame(data    = local_model.predict(x_fit[0]), 
                                                         index   = y_true[0].index, 
                                                         columns = [y_true[0].name]),
                                            pd.DataFrame(data    = local_model.predict(x_fit[1]), 
                                                         index   = y_true[1].index, 
                                                         columns = [y_true[1].name])])
                        R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                                                                    y_pred    = y_fit,
                                                                    y_true    = y_true,
                                                                    display   = False,
                                                                    step_size = step_size,
                                                                    freq      = freq,
                                                                    ax        = None)
                        print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TRAIN}")
                        print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TEST}")
                        local_d[values] = local_d.get(values, 0) + R2_SCORE_TRAIN + R2_SCORE_TEST
                        print()                    
                    print(f"🌟 Best R2 for {key} = {max(local_d, key=local_d.get)} (Total R2 = {local_d[max(local_d, key=local_d.get)]})")
                    return
                
                local_total = len(params[key])
                for j, values in enumerate(params[key], 1):
                    print(f"Turn: {j}/{local_total}")
                    local_model = deepcopy(model)
                    local_model = local_model.set_params(**{key: values})
                    print(local_model)
                    local_model = local_model.fit(x_fit[0],y_true[0])
                    y_fit       = list([pd.DataFrame(data    = local_model.predict(x_fit[0]), 
                                                     index   = y_true[0].index, 
                                                     columns = [y_true[0].name]),
                                        pd.DataFrame(data    = local_model.predict(x_fit[1]), 
                                                     index   = y_true[1].index, 
                                                     columns = [y_true[1].name])])
                    param_key   = f"{key}_{values}"
                    # Option 2
                    if evaluate_metrics.get("MAE") is not None:
                        MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
                                                                       y_pred    = y_fit,
                                                                       y_true    = y_true,
                                                                       display   = False,
                                                                       step_size = step_size,
                                                                       freq      = freq,
                                                                       ax        = None)
                        # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MAE_SCORE_TRAIN + MAE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MAE_SCORE_TRAIN + MAE_SCORE_TEST
                        print()

                    # Option 3
                    if evaluate_metrics.get("MSE") is not None:
                        MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
                                                                       y_pred    = y_fit,
                                                                       y_true    = y_true,
                                                                       display   = False,
                                                                       step_size = step_size,
                                                                       freq      = freq,
                                                                       ax        = None)
                        # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MSE_SCORE_TRAIN + MSE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MSE_SCORE_TRAIN + MSE_SCORE_TEST
                        print()

                    # Option 4
                    if evaluate_metrics.get("MSLE") is not None:
                        MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
                                                                          y_pred    = y_fit,
                                                                          y_true    = y_true,
                                                                          display   = False,
                                                                          step_size = step_size,
                                                                          freq      = freq,
                                                                          ax        = None)
                        # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                        print()

                    # Option 5
                    if evaluate_metrics.get("MAPE") is not None:
                        MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
                                                                          y_pred    = y_fit,
                                                                          y_true    = y_true,
                                                                          display   = False,
                                                                          step_size = step_size,
                                                                          freq      = freq,
                                                                          ax        = None)
                        # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                        print()
            best_param_key = min(global_d, key=global_d.get)
            if global_d[best_param_key] < global_best:
                print(global_d)
                print(f"🌟 Better MAE_MSE_MSLE_MAPE's score params have founded!!!")
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            else:
                print(global_d)
                print(f"🌟 None better MAE_MSE_MSLE_MAPE's score params have founded!!!")
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")

                    # # Option 6
                    # if evaluate_metrics.get("R2") is not None:
                    #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                    #                                                 y_pred    = y_fit,
                    #                                                 y_true    = y_true,
                    #                                                 display   = False,
                    #                                                 freq      = freq,
                    #                                                 ax        = None)
                    #     print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TRAIN}")
                    #     print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TEST}")
                    #     print()
        elif method == "full":        
            fig, axes = plt.subplots(6, 2, figsize=(20, 30))
            
            from scripts.evaluate_model import (My_R2_SCORE,
                                                My_MAE_SCORE,
                                                My_MSE_SCORE,
                                                My_MSLE_SCORE,
                                                My_MAPE_SCORE)
            global_d     = dict({})
            global_total = len(params)
            global_best  = 0
            for i, key in enumerate(params.keys(), 1):
                print(f"Lap: {i}/{global_total}")
                # Option 1
                if evaluate_metrics.get("R2") is not None:
                    local_d = dict({})
                    for values in params[key]:
                        local_model = deepcopy(model)
                        local_model = local_model.set_params(**{key: values})
                        local_model = local_model.fit(x_fit[0],y_true[0])
                        y_fit       = list([pd.DataFrame(data    = local_model.predict(x_fit[0]), 
                                                         index   = y_true[0].index, 
                                                         columns = [y_true[0].name]),
                                            pd.DataFrame(data    = local_model.predict(x_fit[1]), 
                                                        index   = y_true[1].index, 
                                                        columns = [y_true[1].name])])
                        R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                                                                    y_pred    = y_fit,
                                                                    y_true    = y_true,
                                                                    display   = False,
                                                                    step_size = step_size,
                                                                    freq      = freq,
                                                                    ax        = list([axes[0,0],axes[0,1]]))
                        print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TRAIN}")
                        print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TEST}")
                        local_d[values] = R2_SCORE_TRAIN + R2_SCORE_TEST
                        print()                    
                    print(f"🌟 Best R2 for {key} = {max(local_d, key=local_d.get)} (Total R2 = {local_d[max(local_d, key=local_d.get)]})")
                    return                       
                
                local_total = len(params[key])
                for j, values in enumerate(params[key], 1):
                    print(f"Turn: {j}/{local_total}")
                    local_model = deepcopy(model)
                    local_model = local_model.set_params(**{key: values})
                    print(local_model)
                    local_model = local_model.fit(x_fit[0],y_true[0])
                    y_fit       = list([pd.DataFrame(data    = local_model.predict(x_fit[0]), 
                                                     index   = y_true[0].index, 
                                                     columns = [y_true[0].name]),
                                        pd.DataFrame(data    = local_model.predict(x_fit[1]), 
                                                     index   = y_true[1].index, 
                                                     columns = [y_true[1].name])])
                    param_key   = f"{key}_{values}"
                    # Option 2
                    if evaluate_metrics.get("MAE") is not None:
                        MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols  = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        freq      = freq,
                                                                        ax        = list([axes[1,0],axes[1,1]]))
                        # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MAE_SCORE_TRAIN + MAE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MAE_SCORE_TRAIN + MAE_SCORE_TEST
                        print()

                    # Option 3
                    if evaluate_metrics.get("MSE") is not None:
                        MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols  = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        freq      = freq,
                                                                        ax        = list([axes[2,0],axes[2,1]]))
                        # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MSE_SCORE_TRAIN + MSE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MSE_SCORE_TRAIN + MSE_SCORE_TEST
                        print()

                    # Option 4
                    if evaluate_metrics.get("MSLE") is not None:
                        MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols   = target_cols_name,
                                                                            y_pred    = y_fit,
                                                                            y_true    = y_true,
                                                                            display   = False,
                                                                            step_size = step_size,
                                                                            freq      = freq,
                                                                            ax        = list([axes[3,0],axes[3,1]]))
                        # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                        print()

                    # Option 5
                    if evaluate_metrics.get("MAPE") is not None:
                        MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols   = target_cols_name,
                                                                            y_pred    = y_fit,
                                                                            y_true    = y_true,
                                                                            display   = False,
                                                                            step_size = step_size,
                                                                            freq      = freq,
                                                                            ax        = list([axes[4,0],axes[4,1]]))
                        # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TRAIN}")
                        # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TEST}")
                        global_d[param_key] = global_d.get(param_key, 0) + MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                        print(global_d[param_key])
                        if j==1 and i==1:
                            global_best += MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                        print()
            best_param_key = min(global_d, key=global_d.get)
            if global_d[best_param_key] < global_best:
                print(global_d)
                print(f"🌟 Better MAE_MSE_MSLE_MAPE's score params have founded!!!")
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            else:
                print(global_d)
                print(f"🌟 None better MAE_MSE_MSLE_MAPE's score params have founded!!!")
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")

                # # Option 6
                # if evaluate_metrics.get("R2") is not None:
                #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                #                                                 y_pred    = y_fit,
                #                                                 y_true    = y_true,
                #                                                 display   = display,
                #                                                 freq      = freq,
                #                                                 ax        = list([axes[0,0],axes[0,1]]))
                #     print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TRAIN}")
                #     print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TEST}")
                #     print()
              
        else:
            raise ValueError(f"Giá trị method không hợp lệ: {method}")

        if display is True:
            plt.suptitle(f'Evaluation Model - {name}', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            plt.close(fig)

    else:
        raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")


    

