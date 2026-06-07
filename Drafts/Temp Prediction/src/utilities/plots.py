from sklearn.neighbors import LocalOutlierFactor
## Outlier data


def plot_Outlier(data, data_cols, target=None):
    """
    Hiển thị histogram và boxplot cho từng biến số trong data_cols.
    - Nếu có target: hiển thị theo class
    - Nếu không có: hiển thị phân phối và boxplot thông thường
    """
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_cols = len(data_cols)
    ncols    = 2
    nrows    = num_cols

    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = (6 * ncols, 4 * nrows)
    )

    for i, column in enumerate(data_cols):
        # Histplot
        sns.kdeplot(
            data = data,
            x    = column,
            hue  = target if target else None,
            fill = True,
            ax   = axes[i, 0]
        )
        if target:
            axes[i, 0].set_title(f'Histogram: {column} by {target}')
        else:
            axes[i, 0].set_title(f'Histogram: {column}')
        axes[i, 0].grid(True)

        # Boxplot
        if target:
            sns.boxplot(
                data = data,
                x    = column,
                y    = target,
                ax   = axes[i, 1]
            )
            axes[i, 1].set_title(f'Boxplot: {column} by {target}')
            axes[i, 1].set_ylabel(target)
        else:
            sns.boxplot(
                data = data,
                x    = column,
                ax   = axes[i, 1]
            )
            axes[i, 1].set_title(f'Boxplot: {column}')
        # axes[i, 1].set_xlabel(column)
        axes[i, 1].grid(True)

    os.makedirs("../edas/boxplot", exist_ok=True)
    plt.savefig(f"../edas/boxplot/outlier_plot.png", dpi=300, bbox_inches='tight')

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
    
    import sys
    import os
    sys.path.append(os.path.abspath("../src"))
    
    from src.utilities.dataset import HandleMissing_interpolate
    
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
                df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                        method = "time").reset_index()


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
            df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                    method = "time").reset_index()


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
                                    z_thresh          = 3,    # Z_score
                                    modified_z_thresh = 3.5,  # Z_score modified
                                    k                 = 1.5,  # IQR
                                    p_low             = 0.01, # Percentile
                                    p_high            = 0.99, # Percentile
                                    models            = dict({"LocalOutlierFactor" : LocalOutlierFactor()}),
                                    metrics           = list(["z_score"]),
                                    factor            = 1.5,
                                    step_size         = 10,
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
    
    from src.utilities.dataset import HandleMissing_interpolate


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
            df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                    method = "time").reset_index()

        df_filtered = df_filtered.set_index('time')

        for feature in data_cols:
            if method == "statistic":
                fig, axes = plt.subplots(4, 2, figsize=(20, 20))
                
                from models.anomaly_models import (MyZ_Score,
                                                   MyZ_Score_modified,
                                                   MyIQR,
                                                   MyPercentile)
                
                # Option 1
                if "z_score" in metrics:
                    Z_outlier = MyZ_Score(data      = df_filtered,
                                            data_cols = feature,
                                            display   = display,
                                            z_thresh  = z_thresh,
                                            ax        = list([axes[0,0], axes[0,1]]))
                    print(f"🔹 {feature} (Z_Score, z_thresh={z_thresh}): {len(Z_outlier)} outliers ~ {len(Z_outlier)/len(df_filtered[feature]):.2%}")

                # Option 2
                if "z_score modified" in metrics:
                    ZM_outlier = MyZ_Score_modified(data              = df_filtered,
                                                    data_cols         = feature,
                                                    display           = display,
                                                    modified_z_thresh = modified_z_thresh,
                                                    ax                = list([axes[1,0], axes[1,1]]))
                    print(f"🔹 {feature} (Z_Score_Modified, modified_z_thresh={modified_z_thresh}): {len(ZM_outlier)} outliers ~ {len(ZM_outlier)/len(df_filtered[feature]):.2%}")

                # Option 3
                if "iqr" in metrics:
                    IQR_outlier = MyIQR(data      = df_filtered,
                                        data_cols = feature,
                                        display   = display,
                                        k         = k,
                                        ax        = list([axes[2,0], axes[2,1]]))
                    print(f"🔹 {feature} (IQR, k={k}): {len(IQR_outlier)} outliers ~ {len(IQR_outlier)/len(df_filtered[feature]):.2%}")

                # Option 4
                if "percentile" in metrics:
                    Pe_outlier = MyPercentile(data      = df_filtered,
                                              data_cols = feature,
                                              display   = display,
                                              p_low     = p_low,
                                              p_high    = p_high,
                                              ax        = list([axes[3,0], axes[3,1]]))
                    print(f"🔹 {feature} (Percentile, p_low, p_high={p_low, p_high}): {len(Pe_outlier)} outliers ~ {len(Pe_outlier)/len(df_filtered[feature]):.2%}")
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
                                                            step_size   = step_size,
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
def make_mi_scores(features, target, random_state, type):
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    
    features = features.copy()
    for colname in features.select_dtypes(["object", "category"]):
        features[colname], _ = features[colname].factorize()

    discrete_features = []
    for col in features.columns:
        if pd.api.types.is_integer_dtype(features[col]):
            # ngưỡng 20 unique coi là discrete, bạn có thể chỉnh
            discrete_features.append(features[col].nunique() < 20)
        else:
            discrete_features.append(False)
            
    if type =="classification":
    # All discrete features should now have integer dtypes
        mi_scores = mutual_info_classif(features, target, 
                                        discrete_features = discrete_features, 
                                        random_state      = random_state)
    elif type =="regression":
        mi_scores = mutual_info_regression(features, target, 
                                           discrete_features = discrete_features, 
                                           random_state      = random_state)
    else:
        print("Type not support")
        return
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores, label=None, ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Sắp xếp tăng dần để vẽ barh (thanh ngang)
    scores = scores.sort_values(ascending=True)
    # plt.figure(figsize=(16, 9))
    plt.grid(False)

    # # 🏷️ Tiêu đề
    # plt.title("MỨC ĐỘ TƯƠNG QUAN THEO THÔNG TIN TƯƠNG HỖ",
    #           fontweight="bold", va="center", pad=20, fontsize=20)

    # 🎯 Giảm độ dày cột
    if ax is None:
        ax = sns.barplot(x=scores.values, y=scores.index, width=0.5, color="#1f77b4", label=label)
    else:
        sns.barplot(x=scores.values, y=scores.index, width=0.5, color="#1f77b4", label=label, ax=ax)

    # 📈 Đưa trục X lên trên
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # 🧭 Nhãn trục
    # ax.set_xlabel("Giá trị MI (THÔNG TIN TƯƠNG HỖ)", fontweight="bold", fontsize=16)
    ax.set_ylabel("Đặc trưng", fontweight="bold", fontsize=16)

    # ✂️ Chỉ giữ lại viền trên
    for spine in ['left', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(1.2)

    # 🧾 Ghi giá trị MI bên phải cột
    max_val = scores.max()
    for i, v in enumerate(scores.values):
        ax.text(
            x=v + (max_val * 0.02),
            y=i,
            s=f"{v:.3f}",
            ha='left',
            va='center',
            fontsize=16
        )

    # 🔠 Kích thước chữ
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    plt.tight_layout()
    # plt.show()
    return ax

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
    
    from src.utilities.dataset import HandleMissing_interpolate


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
            df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                    method = "time").reset_index()

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
# def plot_evaluate_model_over_time(
#                                 #   data, 
#                                   target_cols_name, station_name, y_true, y_pred,
#                                   method           = "short",
#                                   metrics    = list([
#                                                      "R2",   
#                                                     #  "MAE",
#                                                     #  "MSE",
#                                                     #  "MSLE",
#                                                     #  "MAPE"
#                                                      ]),
#                                   display    = False,
#                                   start_time = None,
#                                   end_time   = None,
#                                   step_size  = 24,
#                                   freq       = None):
#     import seaborn as sns
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
#     from scipy.stats import median_abs_deviation
#     from copy import deepcopy
#     import numpy as np
#     import sys
#     import os
#     sys.path.append(os.path.abspath("../src"))
    
#     from src.utilities.dataset import HandleMissing_interpolate
    
#     # if isinstance(data, pd.DataFrame):
#     name = station_name if station_name else "Unknown"
#     print(f"🔸 Trạm: {name}")

#     #     start_time_ = pd.to_datetime(start_time) if start_time else None
#     #     end_time_   = pd.to_datetime(end_time)   if end_time else None
#     #     df_filtered = data.copy()
#     #     df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')
#     #     df_filtered = df_filtered.dropna(subset=['time'])

#     #     if start_time_ is not None:
#     #         df_filtered = df_filtered[df_filtered['time'] >= start_time_]
#     #     if end_time_ is not None:
#     #         df_filtered = df_filtered[df_filtered['time'] <= end_time_]

#     #     if freq:
#     #         df_filtered  = df_filtered.set_index('time')
#     #         numeric_cols = df_filtered.select_dtypes(include='number').columns
#     #         df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
#     #                                                 method = "time").reset_index()

#     #     df_filtered = df_filtered.set_index('time')

#     if method == "short":
#         from scripts.evaluate_model import (My_R2_SCORE,
#                                             My_MAE_SCORE,
#                                             My_MSE_SCORE,
#                                             My_MSLE_SCORE,
#                                             My_MAPE_SCORE)
#         # Option 1
#         if "R2" in metrics:
#             R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
#                                                         y_pred    = y_pred,
#                                                         y_true    = y_true_temp,
#                                                         display   = False,
#                                                         step_size = step_size,
#                                                         freq      = freq,
#                                                         ax        = None)
#             print(f"🔹 {target_cols_name}_{name} (R2_train) : {R2_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (R2_test)  : {R2_SCORE_TEST}")
#             print()                    
        
#         # Option 2
#         if "MAE" in metrics:
#             MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
#                                                             y_pred    = y_pred,
#                                                             y_true    = y_true_temp,
#                                                             display   = False,
#                                                             step_size = step_size,
#                                                             freq      = freq,
#                                                             ax        = None)
#             print(f"🔹 {target_cols_name}_{name} (MAE_train) : {MAE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MAE_test)  : {MAE_SCORE_TEST}")
#             print()
            
#         # Option 3
#         if "MSE" in metrics:
#             MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
#                                                             y_pred    = y_pred,
#                                                             y_true    = y_true_temp,
#                                                             display   = False,
#                                                             step_size = step_size,
#                                                             freq      = freq,
#                                                             ax        = None)
#             print(f"🔹 {target_cols_name}_{name} (MSE_train) : {MSE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MSE_test)  : {MSE_SCORE_TEST}")
#             print()
            
#         # Option 4
#         if "MSLE" in metrics:
#             MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
#                                                                 y_pred    = y_pred,
#                                                                 y_true    = y_true_temp,
#                                                                 display   = False,
#                                                                 step_size = step_size,
#                                                                 freq      = freq,
#                                                                 ax        = None)
#             print(f"🔹 {target_cols_name}_{name} (MSLE_train) : {MSLE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MSLE_test)  : {MSLE_SCORE_TEST}")
#             print()
            
#         # Option 5
#         if "MAPE" in metrics:
#             MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
#                                                                 y_pred    = y_pred,
#                                                                 y_true    = y_true_temp,
#                                                                 display   = False,
#                                                                 step_size = step_size,
#                                                                 freq      = freq,
#                                                                 ax        = None)
#             print(f"🔹 {target_cols_name}_{name} (MAPE_train) : {MAPE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MAPE_test)  : {MAPE_SCORE_TEST}")
#             print()
            
#         # # Option 6
#         # if metrics.get("R2") is not None:
#         #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
#         #                                                 y_pred    = y_pred,
#         #                                                 y_true    = y_true_temp,
#         #                                                 display   = False,
#         #                                                 freq      = freq,
#         #                                                 ax        = None)
#         #     print(f"🔹 {target_cols_name}_{name} (R2_train): {R2_SCORE_TRAIN}")
#         #     print(f"🔹 {target_cols_name}_{name} (R2_test): {R2_SCORE_TEST}")
#         #     print()
#     elif method == "full":        
#         fig, axes = plt.subplots(6, 2, figsize=(20, 30))
        
#         from scripts.evaluate_model import (My_R2_SCORE,
#                                             My_MAE_SCORE,
#                                             My_MSE_SCORE,
#                                             My_MSLE_SCORE,
#                                             My_MAPE_SCORE)
#         # Option 1
#         if "R2" in metrics:
#             R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
#                                                         y_pred    = y_pred,
#                                                         y_true    = y_true_temp,
#                                                         display   = display,
#                                                         step_size = step_size,
#                                                         freq      = freq,
#                                                         ax        = list([axes[0,0],axes[0,1]]))
#             print(f"🔹 {target_cols_name}_{name} (R2_train) : {R2_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (R2_test)  : {R2_SCORE_TEST}")
#             print()                    
        
#         # Option 2
#         if "MAE" in metrics:
#             MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
#                                                             y_pred    = y_pred,
#                                                             y_true    = y_true_temp,
#                                                             display   = display,
#                                                             step_size = step_size,
#                                                             freq      = freq,
#                                                             ax        = list([axes[1,0],axes[1,1]]))
#             print(f"🔹 {target_cols_name}_{name} (MAE_train) : {MAE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MAE_test)  : {MAE_SCORE_TEST}")
#             print()
            
#         # Option 3
#         if "MSE" in metrics:
#             MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
#                                                             y_pred    = y_pred,
#                                                             y_true    = y_true_temp,
#                                                             display   = display,
#                                                             step_size = step_size,
#                                                             freq      = freq,
#                                                             ax        = list([axes[2,0],axes[2,1]]))
#             print(f"🔹 {target_cols_name}_{name} (MSE_train) : {MSE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MSE_test)  : {MSE_SCORE_TEST}")
#             print()
            
#         # Option 4
#         if "MSLE" in metrics:
#             MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
#                                                                 y_pred    = y_pred,
#                                                                 y_true    = y_true_temp,
#                                                                 display   = display,
#                                                                 step_size = step_size,
#                                                                 freq      = freq,
#                                                                 ax        = list([axes[3,0],axes[3,1]]))
#             print(f"🔹 {target_cols_name}_{name} (MSLE_train) : {MSLE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MSLE_test)  : {MSLE_SCORE_TEST}")
#             print()
            
#         # Option 5
#         if "MAPE" in metrics:
#             MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
#                                                                 y_pred    = y_pred,
#                                                                 y_true    = y_true_temp,
#                                                                 display   = display,
#                                                                 step_size = step_size,
#                                                                 freq      = freq,
#                                                                 ax        = list([axes[4,0],axes[4,1]]))
#             print(f"🔹 {target_cols_name}_{name} (MAPE_train) : {MAPE_SCORE_TRAIN}")
#             print(f"🔹 {target_cols_name}_{name} (MAPE_test)  : {MAPE_SCORE_TEST}")
#             print()
            
#         # # Option 6
#         # if metrics.get("R2") is not None:
#         #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
#         #                                                 y_pred    = y_pred,
#         #                                                 y_true    = y_true_temp,
#         #                                                 display   = display,
#         #                                                 freq      = freq,
#         #                                                 ax        = list([axes[0,0],axes[0,1]]))
#         #     print(f"🔹 {target_cols_name}_{name} (R2_train): {R2_SCORE_TRAIN}")
#         #     print(f"🔹 {target_cols_name}_{name} (R2_test): {R2_SCORE_TEST}")
#         #     print()
            
#     else:
#         raise ValueError(f"Giá trị method không hợp lệ: {method}")

#     if display is True:
#         plt.suptitle(f'Evaluation Model - {name}', fontsize=18)
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.show()
#     else:
#         plt.close(fig)

    # else:
    #     raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")
def plot_evaluate_params_over_time(
                                #    data, 
                                   target_cols_name, station_name, x_fit, y_true, model, params, 
                                   scaler     = None,
                                   method     = "short",
                                   type       = "ML",
                                   metrics    = list([
                                                     #  "R2",   
                                                      "MAE",
                                                     #  "MSE",
                                                     #  "MSLE",
                                                     #  "MAPE"
                                                      ]),
                                   display    = False,
                                   start_time = None,
                                   end_time   = None,
                                   step_size  = 24,
                                   record     = None,
                                   freq       = None):
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
    
    from src.utilities.dataset import HandleMissing_interpolate
    
    # if isinstance(data, pd.DataFrame):
    name = station_name if station_name else "Unknown"
    print(f"🔸 Trạm: {name}")

    #     start_time_ = pd.to_datetime(start_time) if start_time else None
    #     end_time_   = pd.to_datetime(end_time)   if end_time else None
    #     df_filtered = data.copy()
    #     df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')
    #     df_filtered = df_filtered.dropna(subset=['time'])

    #     if start_time_ is not None:
    #         df_filtered = df_filtered[df_filtered['time'] >= start_time_]
    #     if end_time_ is not None:
    #         df_filtered = df_filtered[df_filtered['time'] <= end_time_]

    #     if freq:
    #         df_filtered  = df_filtered.set_index('time')
    #         numeric_cols = df_filtered.select_dtypes(include='number').columns
    #         df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
    #                                                 method = "time").reset_index()

    #     df_filtered = df_filtered.set_index('time')

    # Set defaul params for model
    model.set_params(**dict({k : v[0] for k,v in params.items()}))
    print("default_model: ", model)
    
    if method == "short":
        from scripts.evaluate_model import (My_R2_SCORE,
                                            My_MAE_SCORE,
                                            My_MSE_SCORE,
                                            My_RMSE_SCORE,
                                            My_MSLE_SCORE,
                                            My_MAPE_SCORE)
        global_d     = dict({})
        global_total = len(params)
        global_best  = 0
        for i, key in enumerate(params.keys(), 1):
            print(f"Lap: {i}/{global_total}")
            # Option 1
            if "R2" in metrics:
                local_d = dict({})
                for values in params[key]:
                    local_model = deepcopy(model)
                    local_model = local_model.set_params(**{key: values})
                    local_model = local_model.fit(x_fit[0], y_true[0],
                                                    x_fit[1], y_true[1])
                    if type == "ML":
                        y_fit  = list([pd.DataFrame(data    = local_model.predict_history(type="train"), 
                                                    index   = y_true[0].index[local_model.get_params()["input_chunk_length"]:], 
                                                    columns = [y_true[0].name]),
                                        pd.DataFrame(data    = local_model.predict_history(type="valid"), 
                                                     index   = y_true[1].index, 
                                                     columns = [y_true[1].name])])
                        y_true = list([y_true[0].iloc[local_model.get_params()["input_chunk_length"]:], y_true[1]])
                    if type == "DL":
                        y_fit  = list([pd.DataFrame(data    = local_model.predict_history(type="train"), 
                                                    index   = y_true[0].index[local_model.get_params()["input_chunk_length"]:], 
                                                    columns = [y_true[0].name]),
                                        pd.DataFrame(data    = local_model.predict_history(type="valid"), 
                                                     index   = y_true[1].index, 
                                                     columns = [y_true[1].name])])
                        y_true = list([y_true[0].iloc[local_model.get_params()["input_chunk_length"]:], y_true[1]])
                    R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                                                                y_pred    = y_fit,
                                                                y_true    = y_true_temp,
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
                if not ((i==1 and j==1) or (j!=1)):
                    print("Skip")
                    continue
                print(local_model)
                local_model = local_model.fit(x_fit[0], y_true[0],
                                              x_fit[1], y_true[1])
                
                total_score = 0
                
                if type == "ML":
                        y_fit  = list([pd.DataFrame(data    = local_model.predict_history(type="train"), 
                                                    index   = y_true[0].index[local_model.get_params()["input_chunk_length"]:], 
                                                    columns = [y_true[0].name]),
                                        pd.DataFrame(data    = local_model.predict_history(type="valid"), 
                                                     index   = y_true[1].index, 
                                                     columns = [y_true[1].name])])
                        y_true_temp = list([y_true[0].iloc[local_model.get_params()["input_chunk_length"]:], y_true[1]])
                if type == "DL":
                    y_fit  = list([pd.DataFrame(data    = local_model.predict_history(type="train"), 
                                                index   = y_true[0].index[local_model.get_params()["input_chunk_length"]:], 
                                                columns = [y_true[0].name]),
                                    pd.DataFrame(data    = local_model.predict_history(type="valid"), 
                                                    index   = y_true[1].index, 
                                                    columns = [y_true[1].name])])
                    y_true_temp = list([y_true[0].iloc[local_model.get_params()["input_chunk_length"]:], y_true[1]])
                param_key   = f"{key}_{values}"
                # Option 2
                if "MAE" in metrics:
                    # try:
                    MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols = target_cols_name,
                                                                    y_pred    = y_fit,
                                                                    y_true    = y_true_temp,
                                                                    display   = False,
                                                                    step_size = step_size,
                                                                    scaler    = scaler,
                                                                    freq      = freq,
                                                                    ax        = None)
                    total_score += MAE_SCORE_TEST
                    print(total_score)
                    if j==1 and i==1:
                        global_best += MAE_SCORE_TEST
                    # except Exception as e:
                    #     print(f"Something went wrong MAE... SKip this params {params[key]}")
                    #     continue
                    print()

                # Option 3
                if "MSE" in metrics:
                # try:
                    MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols = target_cols_name,
                                                                    y_pred    = y_fit,
                                                                    y_true    = y_true_temp,
                                                                    display   = False,
                                                                    step_size = step_size,
                                                                    scaler    = scaler,
                                                                    freq      = freq,
                                                                    ax        = None)
                    total_score += MSE_SCORE_TEST
                    print(total_score)
                    if j==1 and i==1:
                        global_best += MSE_SCORE_TEST
                # except Exception as e:
                #     print(f"Something went wrong MSE... SKip this params {params[key]}")
                #     continue
                    print()

                # Option 3.2
                if "RMSE" in metrics:
                # try:
                    RMSE_SCORE_TRAIN, RMSE_SCORE_TEST = My_RMSE_SCORE(data_cols  = target_cols_name,
                                                                    y_pred    = y_fit,
                                                                    y_true    = y_true_temp,
                                                                    display   = False,
                                                                    step_size = step_size,
                                                                    scaler    = scaler,
                                                                    freq      = freq,
                                                                    ax        = None)
                    total_score += RMSE_SCORE_TEST
                    print(total_score)
                    if j==1 and i==1:
                        global_best += RMSE_SCORE_TEST
                # except Exception as e:
                #     print(f"Something went wrong RMSE... SKip this params {params[key]}")
                #     continue
                    print()

                # Option 4
                if "MSLE" in metrics:
                # try:
                    MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true_temp,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        scaler    = scaler,
                                                                        freq      = freq,
                                                                        ax        = None)
                    total_score += MSLE_SCORE_TEST
                    print(total_score)
                    if j==1 and i==1:
                        global_best += MSLE_SCORE_TEST
                # except Exception as e:
                #     print(f"Something went wrong MSLE... SKip this params {params[key]}")
                #     continue
                    print()

                # Option 5
                if "MAPE" in metrics:
                # try:
                    MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true_temp,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        scaler    = scaler,
                                                                        freq      = freq,
                                                                        ax        = None)
                    total_score += MAPE_SCORE_TEST
                    print(total_score)
                    if j==1 and i==1:
                        global_best += MAPE_SCORE_TEST
                # except Exception as e:
                #     print(f"Something went wrong... SKip this params {params[key]}")
                #     continue
                    
                global_d[param_key] = total_score
                print()
                    
                # Option 6
        best_param_key = min(global_d, key=global_d.get)
        if global_d[best_param_key] < global_best:
            print(global_d)
            print(f"🌟 Better MAE_MSE_MSLE_MAPE's score params have founded!!!")
            print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            if record is not None:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {record})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")
            else:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")
        else:
            print(global_d)
            print(f"🌟 None better MAE_MSE_MSLE_MAPE's score params have founded!!!")
            print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            if record is not None:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {record})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/record):.2%}")
            else:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")

                # # Option 6
                # if metrics.get("R2") is not None:
                #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
                #                                                 y_pred    = y_fit,
                #                                                 y_true    = y_true_temp,
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
                                            My_RMSE_SCORE,
                                            My_MSLE_SCORE,
                                            My_MAPE_SCORE)
        global_d     = dict({})
        global_total = len(params)
        global_best  = 0
        for i, key in enumerate(params.keys(), 1):
            print(f"Lap: {i}/{global_total}")
            # Option 1
            if "R2" in metrics:
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
                                                                y_true    = y_true_temp,
                                                                display   = False,
                                                                step_size = step_size,
                                                                freq      = freq,
                                                                ax        = list([axes[0,0],axes[0,1]]))
                    print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TRAIN}")
                    print(f"🔹 {target_cols_name}_{name} (R2_{key} = {values} : {R2_SCORE_TEST}")
                    # local_d[values] = R2_SCORE_TRAIN + R2_SCORE_TEST
                    local_d[values] = R2_SCORE_TRAIN
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
                if "MAE" in metrics:
                    try:
                        MAE_SCORE_TRAIN, MAE_SCORE_TEST = My_MAE_SCORE(data_cols  = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true_temp,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        freq      = freq,
                                                                        ax        = list([axes[1,0],axes[1,1]]))
                    except Exception as e:
                        print(f"Something went wrong... SKip this params {params[key]}")
                        continue
                    # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TRAIN}")
                    # print(f"🔹 {target_cols_name}_{name} (MAE_{key} = {values} : {MAE_SCORE_TEST}")
                    # global_d[param_key] = global_d.get(param_key, 0) + MAE_SCORE_TRAIN + MAE_SCORE_TEST
                    global_d[param_key] = global_d.get(param_key, 0) + MAE_SCORE_TEST
                    print(global_d[param_key])
                    if j==1 and i==1:
                        # global_best += MAE_SCORE_TRAIN + MAE_SCORE_TEST
                        global_best += MAE_SCORE_TEST
                    print()

                # Option 3
                if "MSE" in metrics:
                    try:
                        MSE_SCORE_TRAIN, MSE_SCORE_TEST = My_MSE_SCORE(data_cols  = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true_temp,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        freq      = freq,
                                                                        ax        = list([axes[2,0],axes[2,1]]))
                    except Exception as e:
                        print(f"Something went wrong... SKip this params {params[key]}")
                        continue
                    # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TRAIN}")
                    # print(f"🔹 {target_cols_name}_{name} (MSE_{key} = {values} : {MSE_SCORE_TEST}")
                    # global_d[param_key] = global_d.get(param_key, 0) + MSE_SCORE_TRAIN + MSE_SCORE_TEST
                    global_d[param_key] = global_d.get(param_key, 0) + MSE_SCORE_TEST
                    print(global_d[param_key])
                    if j==1 and i==1:
                        # global_best += MSE_SCORE_TRAIN + MSE_SCORE_TEST
                        global_best += MSE_SCORE_TEST
                    print()
                    
                # Option 3.2
                if "RMSE" in metrics:
                    try:
                        RMSE_SCORE_TRAIN, RMSE_SCORE_TEST = My_RMSE_SCORE(data_cols  = target_cols_name,
                                                                        y_pred    = y_fit,
                                                                        y_true    = y_true_temp,
                                                                        display   = False,
                                                                        step_size = step_size,
                                                                        freq      = freq,
                                                                        ax        = list([axes[2,0],axes[2,1]]))
                    except Exception as e:
                        print(f"Something went wrong... SKip this params {params[key]}")
                        continue
                    # print(f"🔹 {target_cols_name}_{name} (RMSE_{key} = {values} : {RMSE_SCORE_TRAIN}")
                    # print(f"🔹 {target_cols_name}_{name} (RMSE_{key} = {values} : {RMSE_SCORE_TEST}")
                    # global_d[param_key] = global_d.get(param_key, 0) + RMSE_SCORE_TRAIN + RMSE_SCORE_TEST
                    global_d[param_key] = global_d.get(param_key, 0) + RMSE_SCORE_TEST
                    print(global_d[param_key])
                    if j==1 and i==1:
                        # global_best += RMSE_SCORE_TRAIN + RMSE_SCORE_TEST
                        global_best += RMSE_SCORE_TEST
                    print()

                # Option 4
                if "MSLE" in metrics:
                    try:
                        MSLE_SCORE_TRAIN, MSLE_SCORE_TEST = My_MSLE_SCORE(data_cols   = target_cols_name,
                                                                            y_pred    = y_fit,
                                                                            y_true    = y_true_temp,
                                                                            display   = False,
                                                                            step_size = step_size,
                                                                            freq      = freq,
                                                                            ax        = list([axes[3,0],axes[3,1]]))
                    except Exception as e:
                        print(f"Something went wrong... SKip this params {params[key]}")
                        continue
                    # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TRAIN}")
                    # print(f"🔹 {target_cols_name}_{name} (MSLE_{key} = {values} : {MSLE_SCORE_TEST}")
                    # global_d[param_key] = global_d.get(param_key, 0) + MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                    global_d[param_key] = global_d.get(param_key, 0) + MSLE_SCORE_TEST
                    print(global_d[param_key])
                    if j==1 and i==1:
                        # global_best += MSLE_SCORE_TRAIN + MSLE_SCORE_TEST
                        global_best += MSLE_SCORE_TEST
                    print()

                # Option 5
                if "MAPE" in metrics:
                    try:
                        MAPE_SCORE_TRAIN, MAPE_SCORE_TEST = My_MAPE_SCORE(data_cols   = target_cols_name,
                                                                            y_pred    = y_fit,
                                                                            y_true    = y_true_temp,
                                                                            display   = False,
                                                                            step_size = step_size,
                                                                            freq      = freq,
                                                                            ax        = list([axes[4,0],axes[4,1]]))
                    except Exception as e:
                        print(f"Something went wrong... SKip this params {params[key]}")
                        continue
                    # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TRAIN}")
                    # print(f"🔹 {target_cols_name}_{name} (MAPE_{key} = {values} : {MAPE_SCORE_TEST}")
                    # global_d[param_key] = global_d.get(param_key, 0) + MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                    global_d[param_key] = global_d.get(param_key, 0) + MAPE_SCORE_TEST
                    print(global_d[param_key])
                    if j==1 and i==1:
                        # global_best += MAPE_SCORE_TRAIN + MAPE_SCORE_TEST
                        global_best += MAPE_SCORE_TEST
                    print()
        best_param_key = min(global_d, key=global_d.get)
        if global_d[best_param_key] < global_best:
            print(global_d)
            print(f"🌟 Better MAE_MSE_MSLE_MAPE's score params have founded!!!")
            print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            if record is not None:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {record})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")
            else:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")
        else:
            print(global_d)
            print(f"🌟 None better MAE_MSE_MSLE_MAPE's score params have founded!!!")
            print(f"🌟 NEW Best MAE_MSE_MSLE_MAPE = {best_param_key} (Total = {global_d[best_param_key]})")
            if record is not None:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {record})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/record):.2%}")
            else:
                print(f"🌟 OLD Best MAE_MSE_MSLE_MAPE (Total = {global_best})")
                print(f"🌟 Improve Ratio = {(1 - global_d[best_param_key]/global_best):.2%}")

            # # Option 6
            # if metrics.get("R2") is not None:
            #     R2_SCORE_TRAIN, R2_SCORE_TEST = My_R2_SCORE(data_cols = target_cols_name,
            #                                                 y_pred    = y_fit,
            #                                                 y_true    = y_true_temp,
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

    # else:
    #     raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")

def plot_periodogram(ts, detrend='linear', ax=None, label=None, color=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import periodogram

    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(ts, fs=fs)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 6))

    ax.step(freqencies, spectrum, color=color or "purple", label=label)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    
    return ax
  
def seasonal_plot(data, data_cols, period, freq, 
                  display = False,
                  ax      = None):
    import seaborn as sns
    X = data.copy()
    
    X["dayofweek"] = X.index.dayofweek  # the x-axis (freq)# days within a week
    X["week"] = X.index.isocalendar().week  # the seasonal period (period)
    X["dayofmonth"] = X.index.day  # the x-axis (freq)
    X["month"] = X.index.month  # the seasonal period (period)
    # days within a year
    X["dayofyear"] = X.index.dayofyear
    X["year"] = X.index.year

    # Semiannual: 2 kỳ/năm → 1–2
    X["semiannual"] = (X.index.month - 1) // 6 + 1

    # Quarterly: 4 kỳ/năm → 1–4
    X["quarterly"] = (X.index.month - 1) // 3 + 1

    # Bimonthly: 6 kỳ/năm → 1–6
    X["bimonthly"] = (X.index.month - 1) // 2 + 1

    # Biweekly: 26 kỳ/năm → tuần lẻ/chu kỳ 2 tuần
    X["biweekly"] = (X.index.isocalendar().week - 1) // 2 + 1

    # Semiweekly: 104 kỳ/năm → 1 kỳ = nửa tuần → tính theo ngày trong tuần
    X["semiweekly"] = (X.index.dayofyear - 1) // (365/104) + 1
    X["semiweekly"] = X["semiweekly"].astype(int)  # chuyển sang int
    
    y = data_cols

    if display is True:        
        palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
        ax = sns.lineplot(
            x       = freq,
            y       = y,
            hue     = period,
            data    = X,
            ci      = False,
            ax      = ax,
            palette = palette,
            legend  = False,
        )
        ax.set_title(f"Seasonal Plot ({period}/{freq})")
        for line, name in zip(ax.lines, X[period].unique()):
            y_ = line.get_ydata()[-1]
            ax.annotate(
                name,
                xy         = (1, y_),
                xytext     = (6, 0),
                color      = line.get_color(),
                xycoords   = ax.get_yaxis_transform(),
                textcoords = "offset points",
                size       = 14,
                va         = "center",
            )
        return ax

def seasonal_forecast(data, data_cols):
    import pandas as pd
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
    from sklearn.linear_model import LinearRegression

    df = data.asfreq("D")   
    fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality
    dp = DeterministicProcess(
        index            = df.index,
        constant         = True,               # dummy feature for bias (y-intercept)
        order            = 1,                     # trend (order 1 means linear)
        seasonal         = False,               # weekly seasonality (indicators)
        additional_terms = [fourier],  # annual seasonality (fourier)
        drop             = True,                   # drop terms to avoid collinearity
    )
    X_dp = dp.in_sample()  # create features for dates in tunnel.index    
    y = df[data_cols]
    model = LinearRegression(fit_intercept=False)
    _ = model.fit(X_dp, y)
    y_pred = pd.Series(model.predict(X_dp), index=y.index)
    X_fore = dp.out_of_sample(steps=90)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
    return y, y_pred, y_fore

def seasonal_anlysis(data, data_cols,
                    method     = "short",
                    seasonal   = list([
                                    "weekly", 
                                    "monthly", 
                                    "yearly"
                                    ]),
                    display    = False,
                    start_time = None, 
                    end_time   = None, 
                    freq       = None):   
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    import os
    sys.path.append(os.path.abspath("../src"))
    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")

    for column in data_cols:    
        if method == "short":
            # Option 1
            if "Annual" in seasonal: 
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "year",
                              display   = False,
                              ax        = None)           
            # Option 2
            if "Semiannual" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "semiannual",
                              display   = False,
                              ax        = None)
            # Option 3
            if "Quarterly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "quarterly",
                              display   = False,
                              ax        = None)
            # Option 4
            if "Bimonthly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofmonth",
                              period    = "bimonthly",
                              display   = False,
                              ax        = None)
            # Option 5
            if "Monthly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofmonth",
                              period    = "month",
                              display   = False,
                              ax        = None)
            # Option 6
            if "Biweekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "biweekly",
                              display   = False,
                              ax        = None)
            # Option 7
            if "Weekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "week",
                              display   = False,
                              ax        = None)
            # Option 8
            if "Semiweekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "semiweekly",
                              display   = False,
                              ax        = None)
                
        elif method == "full":        
            fig, ax = plt.subplots(4, 2, figsize=(16, 10))

            # Option 1
            if "Annual" in seasonal: 
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "year",
                              display   = display,
                              ax        = ax[0,0])           
            # Option 2
            if "Semiannual" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "semiannual",
                              display   = display,
                              ax        = ax[0,1])
            # Option 3
            if "Quarterly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofyear",
                              period    = "quarterly",
                              display   = display,
                              ax        = ax[1,0])
            # Option 4
            if "Bimonthly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofmonth",
                              period    = "bimonthly",
                              display   = display,
                              ax        = ax[1,1])
            # Option 5
            if "Monthly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofmonth",
                              period    = "month",
                              display   = display,
                              ax        = ax[2,0])
            # Option 6
            if "Biweekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "biweekly",
                              display   = display,
                              ax        = ax[2,1])
            # Option 7
            if "Weekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "week",
                              display   = display,
                              ax        = ax[3,0])
            # Option 8
            if "Semiweekly" in seasonal:
                seasonal_plot(data      = data,
                              data_cols = column,
                              freq      = "dayofweek",
                              period    = "semiweekly",
                              display   = display,
                              ax        = ax[3,1])

def moving_average(data, data_cols, period):
    return data[data_cols].rolling(window      = period,         # 365-day window
                                   center      = True,           # puts the average at the center of the window
                                   min_periods = period // 2,    # choose about half the window size
                                  ).mean()                       # compute the mean (could also do median, std, min, max, ...)

def trend_forecast(data, data_cols):
    import pandas as pd
    from statsmodels.tsa.deterministic import DeterministicProcess
    from sklearn.linear_model import LinearRegression

    df = data.asfreq("D") 
    dp = DeterministicProcess(index    = df.index,     # dates from the training data
                              constant = True,         # dummy feature for the bias (y_intercept)
                              order    = 2,            # the time dummy (trend)
                              drop     = True)         # drop terms if necessary to avoid collinearity
    # `in_sample` creates features for the dates given in the `index` argument
    X = dp.in_sample()
    y = df[data_cols]  # the target
    # The intercept is the same as the `const` feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index)
    X = dp.out_of_sample(steps=30)
    y_fore = pd.Series(model.predict(X), index=X.index)
    return y_pred, y_fore

def trend(data,
          data_cols,
          display   = False,
          period    = None,
          ax        = None):
    import seaborn as sns
    from warnings import simplefilter
    import matplotlib.pyplot as plt

    simplefilter("ignore")  # ignore warnings to clean up output cells
    # Set Matplotlib defaults
    sns.set_theme(style="whitegrid")  # tương đương seaborn-whitegrid
    plt.rc("figure", autolayout=True, figsize=(11, 5))
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        legend=False,
    )

    ma = moving_average(data, data_cols, period)
    y_pred, y_fore = trend_forecast(data, data_cols)

    if display is True:
        data[data_cols].plot(style=".", color="0.5", ax=ax[0])
        ma.plot(linewidth= 3, 
                title    = f"{data_cols} - {period}-Day Moving Average",
                label    = f"{period}-Day MA",
                ax       = ax[0],
                color    = "red")
        ax[0].legend()
        
        data[data_cols].plot(style=".", color="0.5", title=f"{data_cols} - Linear Trend", ax=ax[1])
        y_pred.plot(ax=ax[1], linewidth=3, label="Trend",color="red")
        ax[1].legend()
        
        data[data_cols].plot(ax=ax[2], title=f"{data_cols} - Linear Trend Forecast", **plot_params)
        y_pred.plot(ax=ax[2], linewidth=3, label="Trend")
        y_fore.plot(ax=ax[2], linewidth=3, label="Trend Forecast", color="C3")
        ax[2].legend()

def residuals(data,
              data_cols,
              display   = False,
              period    = None,
              ax        = None):
    from statsmodels.tsa.seasonal import seasonal_decompose
    if display is True:
        for i, model in enumerate(["additive", "multiplicative"]):            
            result = seasonal_decompose(data[data_cols], 
                                        model  = model, 
                                        period = period)            
            ax[i].set_title("Decomposition for " + model + " model - " + data_cols)
            ax[i].plot(result.resid, 'red', label='Residuals')
            ax[i].legend()

def correlation_analysis(data,
                         data_cols = None,
                         display   = False,
                         period    = None,
                         ax        = None):
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    if display is True:
        # acf, pacf
        plot_acf(data[data_cols], lags=period, ax=ax[0])
        plot_pacf(data[data_cols], lags=period, ax=ax[1])

def ts_analysis(data,
                data_cols = None,
                method           = "short",
                analysis    = list([
                                    "trend",  
                                    # "resid",
                                    ]),
                display    = False,
                start_time = None, 
                end_time = None, 
                freq = None,
                period = 365):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    import os
    sys.path.append(os.path.abspath("../src"))
    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")

    for column in data_cols:    
        if method == "short":
            # Option 1
            if "trend" in analysis:
                trend(data      = df_filtered,
                      data_cols = column,
                      display   = False,
                      period    = period,
                      ax        = None)

            # Option 2
            if "resid" in analysis:
                residuals(data      = df_filtered,
                          data_cols = column,
                          display   = False,
                          period    = period,
                          ax        = None)
              
            # Option 3            
              
            # Option 4
              
            # Option 5
              
            # Option 6
            
        elif method == "full":        
            fig, axes = plt.subplots(3, 2, figsize=(20, 20))

            # Option 1
            if "trend" in analysis:
                trend(data      = df_filtered,
                      data_cols = column,
                      display   = display,
                      period    = period,
                      ax        = list([axes[0,0],axes[0,1],axes[1,0]]))
              
            # Option 2
            if "resid" in analysis:
                residuals(data      = df_filtered,
                          data_cols = column,
                          display   = display,
                          period    = period,
                          ax        = list([axes[2,0],axes[2,1]]))
              
            # Option 3            
              
            # Option 4            
              
            # Option 5

            # Option 6

def plot_influence_of_latitude_in_col(data, df_provinces, data_cols,
                                      feature_name = None,
                                      unit         = "°C",
                                      display      = False,
                                      ax           = None):
    import os
    import geopandas as gpd
    import matplotlib.pyplot as plt
    # Tạo GeoDataFrame và gán region trong một bước
    gdf_points = gpd.GeoDataFrame(data     = data,
                                  geometry = gpd.points_from_xy(x = data["LONGITUDE"], 
                                                                y = data["LATITUDE"]),
                                  crs      = "EPSG:4326")
    gdf_points = gdf_points.assign(region=lambda df: df["NAME"].map({"NOI BAI"   : "Đồng bằng sông Hồng",
                                                                     "THANH HOA" : "Bắc Trung Bộ",
                                                                     "DONG HOI"  : "Bắc Trung Bộ",
                                                                     "QUY NHON"  : "Duyên hải Nam Trung Bộ",
                                                                     "TSN"       : "Đông Nam Bộ",
                                                                     "CA MAU"    : "Đồng bằng sông Cửu Long"}))
    # Mapping tỉnh -> vùng
    df_provinces["region"] = df_provinces["ten_tinh"].map({
        **dict.fromkeys(["Hà Nội", "Hải Phòng", "Bắc Ninh", "Hưng Yên", "Quảng Ninh", "Ninh Bình"], 
                         "Đồng bằng sông Hồng"),
        **dict.fromkeys(["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Trị", "Huế"], 
                         "Bắc Trung Bộ"),
        **dict.fromkeys(["Đà Nẵng", "Quảng Ngãi", "Gia Lai", "Khánh Hoà", "Lâm Đồng", "Đắk Lắk"], 
                         "Duyên hải Nam Trung Bộ"),
        **dict.fromkeys(["TP. Hồ Chí Minh", "Đồng Nai", "Tây Ninh"], 
                         "Đông Nam Bộ"),
        **dict.fromkeys(["Cần Thơ", "An Giang", "Cà Mau", "Đồng Tháp", "Vĩnh Long"], 
                         "Đồng bằng sông Cửu Long")})
    
    fig, ax = plt.subplots(figsize=(9, 8))
    # Vẽ tỉnh và vùng
    df_provinces.plot(ax        = ax, 
                      color     = "lightgrey", 
                      edgecolor = "black")
    df_provinces.dissolve(by      = "region", 
                          aggfunc = "first")
    df_provinces = df_provinces.join(gdf_points.groupby("region", observed=True)[data_cols].mean(), on="region")
    df_provinces.plot(ax        = ax,
                      column    = data_cols,
                      cmap      = "coolwarm",
                      edgecolor = "black",
                      linewidth = 1,
                      legend    = True,
                      alpha     = 0.7)
    
    cbar_ax = ax.get_figure().axes[-1]  
    cbar_ax.tick_params(labelsize=12)

    # Vẽ trạm và nhãn 
    for station in gdf_points["NAME"].unique():
        # Lấy dữ liệu trạm và tính trung bình trong cùng vòng lặp
        station_point = gdf_points[gdf_points["NAME"] == station].iloc[0]
        temp_avg      = gdf_points[gdf_points["NAME"] == station][data_cols].mean()
        
        # Vẽ điểm trạm
        ax.scatter(x         = station_point.LONGITUDE, 
                   y         = station_point.LATITUDE, 
                   color     = "green", 
                   s         = 100, 
                   edgecolor = "black")
        
        # Vẽ nhãn
        ax.text(x        = station_point.LONGITUDE + 0.3, 
                y        = station_point.LATITUDE,
                s        = f"{station}\n({temp_avg:.4f}{unit})",
                fontsize = 12,
                ha       = "left",
                va       = "bottom",
                bbox     = dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        
    if display is True:
        ax.set_title(f"Ảnh hưởng của vĩ độ đến {feature_name} tại các trạm khí tượng",
                     fontsize=12, loc="center", weight="bold")
        ax.set_xlabel("Kinh độ", fontsize=12, weight="bold")
        ax.set_ylabel("Vĩ độ", fontsize=12, weight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3)

        os.makedirs("../edas/geochart", exist_ok=True)
        plt.savefig(f"../edas/geochart/influence_of_latitude_in_{feature_name}.png", dpi=300, bbox_inches="tight")
        ax.plot()

def plot_influence_of_latitude_in_features(data, df_provinces,
                                           method       = "short",
                                           features     = list(["Nina_index",
                                                               # "DEW_ave",
                                                               # "TEMP_ave",
                                                               # "RH_ave",
                                                               # "DEW_max",
                                                               # "RH_max",
                                                               # "sp_ave",
                                                               # "tcc_ave",
                                                               # "tp_sum",
                                                               # "wind_speed_ave",
                                                               # "wind_direction_deg_ave",
                                                               # "TEMP_max"
                                                               ]),
                                           display      = False,
                                           start_time   = None, 
                                           end_time     = None,
                                           freq         = None):
    import os
    import sys
    import pandas as pd  
    import matplotlib.pyplot as plt  
    
    sys.path.append(os.path.abspath("../src"))    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")
        
    if method == "short":
        # Option 1
        if "TEMP_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "TEMP_ave",
                                              feature_name = "nhiệt độ trung bình",
                                              unit         = "°C",
                                              display      = False,
                                              # ax           = None
                                              )
            
        # Option 2
        if "TEMP_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "TEMP_max",
                                              feature_name = "nhiệt độ cực đại",
                                              unit         = "°C",
                                              display      = False,
                                              # ax           = None
                                              )
            
        # Option 3
        if "Nina_index" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "Nina_index",
                                              feature_name = "Nina_index",
                                              unit         = "°C",
                                              display      = False,
                                              # ax           = None
                                              )

        # Option 4
        if "DEW_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "DEW_ave",
                                              feature_name = "điểm sương trung bình",
                                              unit         = "°C",
                                              display      = False,
                                              # ax           = None
                                              )  
        # Option 5
        if "RH_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "RH_ave",
                                              feature_name = "độ ẩm trung bình",
                                              unit         = "%",
                                              display      = False,
                                              # ax           = None
                                              )  
        # Option 6
        if "DEW_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "DEW_max",
                                              feature_name = "điểm sương cực đại",
                                              unit         = "°C",
                                              display      = False,
                                              # ax           = None
                                              )

        # Option 7
        if "RH_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "RH_max",
                                              feature_name = "độ ẩm cực đại",
                                              unit         = "%",
                                              display      = False,
                                              # ax           = None
                                              )
        
        # Option 8
        if "sp_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "sp_ave",
                                              feature_name = "áp suất bề mặt trung bình",
                                              unit         = "kPa",
                                              display      = False,
                                              # ax           = None
                                              )
            
        # Option 9
        if "tcc_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "tcc_ave",
                                              feature_name = "lượng mây che phủ trung bình",
                                              unit         = "%",
                                              display      = False,
                                              # ax           = None
                                              )
        
        # Option 10
        if "tp_sum" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "tp_sum",
                                              feature_name = "lượng mưa tích lũy",
                                              unit         = "mm",
                                              display      = False,
                                              # ax           = None
                                              )
        
        # Option 11
        if "ws_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "ws_ave",
                                              feature_name = "tốc độ gió trung bình",
                                              unit         = "m/s",
                                              display      = False,
                                              # ax           = None
                                              )
        
        # Option 12
        if "wd_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "wd_ave",
                                              feature_name = "hướng gió trung bình",
                                              unit         = "°",
                                              display      = False,
                                              # ax           = None
                                              )
        
    elif method == "full":
        # Option 1
        if "TEMP_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "TEMP_ave",
                                              feature_name = "nhiệt độ trung bình",
                                              unit         = "°C",
                                              display      = display,
                                            #   ax           = axes[0]
                                              )

        # Option 2
        if "TEMP_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "TEMP_max",
                                              feature_name = "nhiệt độ cực đại",
                                              unit         = "°C",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )     

        # Option 3
        if "DEW_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "DEW_ave",
                                              feature_name = "điểm sương trung bình",
                                              unit         = "°C",
                                              display      = display,
                                            #   ax           = axes[0]
                                            )  

        # Option 4
        if "DEW_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "DEW_max",
                                              feature_name = "điểm sương cực đại",
                                              unit         = "°C",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )

        # Option 5
        if "RH_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "RH_ave",
                                              feature_name = "độ ẩm tương đối trung bình",
                                              unit         = "%",
                                              display      = display,
                                            #   ax           = axes[0]
                                            )  

        # Option 6
        if "RH_max" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "RH_max",
                                              feature_name = "độ ẩm tương đối cực đại",
                                              unit         = "%",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )

        # Option 7
        if "Nina_index" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "Nina_index",
                                              feature_name = "Nina_index",
                                              unit         = "°C",
                                              display      = display,
                                            #   ax           = axes[0]
                                              )

        # Option 8
        if "sp_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "sp_ave",
                                              feature_name = "áp suất bề mặt trung bình",
                                              unit         = "kPa",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )

        # Option 9
        if "tcc_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "tcc_ave",
                                              feature_name = "lượng mây che phủ trung bình",
                                              unit         = "%",
                                              display      = display,
                                            #   ax           = axes[0]
                                              )

        # Option 10
        if "tp_sum" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "tp_sum",
                                              feature_name = "lượng mưa tích lũy",
                                              unit         = "mm",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )

        # Option 11
        if "ws_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "ws_ave",
                                              feature_name = "tốc độ gió trung bình",
                                              unit         = "m/s",
                                              display      = display,
                                            #   ax           = axes[0]
                                              )

        # Option 12
        if "wd_ave" in features:
            plot_influence_of_latitude_in_col(data         = df_filtered,
                                              df_provinces = df_provinces,
                                              data_cols    = "wd_ave",
                                              feature_name = "hướng gió trung bình",
                                              unit         = "°",
                                              display      = display,
                                            #   ax           = axes[1]
                                              )          

def plot_correlation_matrix_in_station(data, data_cols,
                                       feature_name = None,
                                       display      = False,
                                       ax           = None):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    correlation_matrix = data.pivot_table(index="time", columns="NAME", values=data_cols).corr(method="pearson")
    # Thứ tự cột/hàng bạn muốn
    cols_pos = ["NOI BAI", "THANH HOA", "DONG HOI", "QUY NHON", "TSN", "CA MAU"]

    # Reorder cả index và columns
    correlation_matrix = correlation_matrix.reindex(index   = cols_pos, 
                                                    columns = cols_pos)
    if display is True:
        fig, ax = plt.subplots(figsize=(9, 10))
        sns.heatmap(data     = correlation_matrix, 
                    annot    = True,
                    square   = True, 
                    cmap     = "Blues", 
                    fmt      = ".2f",
                    cbar_kws = {"shrink": 0.7},)
        plt.title(f'Tương quan {feature_name} giữa các trạm khí tượng')
        plt.xlabel("Trạm khí tượng")
        plt.ylabel("Trạm khí tượng")

        os.makedirs("../edas/heatmap", exist_ok=True)
        plt.savefig(f"../edas/heatmap/correlation_matrix_in_{feature_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

def plot_correlation_matrix_in_stations(data, 
                                        method       = "short",
                                        features     = list(["Nina_index",
                                                            # "DEW_ave",
                                                            # "TEMP_ave",
                                                            # "RH_ave",
                                                            # "DEW_max",
                                                            # "RH_max",
                                                            # "sp_ave",
                                                            # "tcc_ave",
                                                            # "tp_sum",
                                                            # "wind_speed_ave",
                                                            # "wind_direction_deg_ave",
                                                            # "TEMP_max"
                                                            ]),
                                        display      = False,
                                        start_time   = None, 
                                        end_time     = None,
                                        freq         = None):
    import os
    import sys
    import pandas as pd    
    
    sys.path.append(os.path.abspath("../src"))    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")
        
    if method == "short":
        # Option 1
        if "TEMP_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "TEMP_ave",
                                               feature_name = "nhiệt độ trung bình",
                                               display      = False,
                                               # ax           = None
                                               )
            
        # Option 2
        if "TEMP_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "TEMP_max",
                                               feature_name = "nhiệt độ cực đại",
                                               display      = False,
                                               # ax           = None
                                               )
            
        # Option 3
        if "Nina_index" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "Nina_index",
                                               feature_name = "Nina_index",
                                               display      = False,
                                               # ax           = None
                                               )

        # Option 4
        if "DEW_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "DEW_ave",
                                               feature_name = "điểm sương trung bình",
                                               display      = False,
                                               # ax           = None
                                               )  
        # Option 5
        if "RH_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "RH_ave",
                                               feature_name = "độ ẩm tương đối trung bình",
                                               display      = False,
                                               # ax           = None
                                               )  
        # Option 6
        if "DEW_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "DEW_max",
                                               feature_name = "điểm sương cực đại",
                                               display      = False,
                                               # ax           = None
                                               )

        # Option 7
        if "RH_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "RH_max",
                                               feature_name = "độ ẩm tương đối cực đại",
                                               display      = False,
                                               # ax           = None
                                               )
        
        # Option 8
        if "sp_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "sp_ave",
                                               feature_name = "áp suất bề mặt trung bình",
                                               display      = False,
                                               # ax           = None
                                               )
            
        # Option 9
        if "tcc_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "tcc_ave",
                                               feature_name = "lượng mây che phủ trung bình",
                                               display      = False,
                                               # ax           = None
                                               )
        
        # Option 10
        if "tp_sum" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "tp_sum",
                                               feature_name = "lượng mưa tích lũy",
                                               display      = False,
                                               # ax           = None
                                               )
        
        # Option 11
        if "ws_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "ws_ave",
                                               feature_name = "tốc độ gió trung bình",
                                               display      = False,
                                               # ax           = None
                                               )
        
        # Option 12
        if "wd_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "wd_ave",
                                               feature_name = "hướng gió trung bình",
                                               display      = False,
                                               # ax           = None
                                               )
        
    elif method == "full":        
        # fig, axes = plt.subplots(6, 2, figsize=(40, 80))            
        # Option 1
        if "TEMP_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "TEMP_ave",
                                               feature_name = "nhiệt độ trung bình",
                                               display      = display,
                                               # ax           = axes[1,0]
                                               )
            
        # Option 2
        if "TEMP_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "TEMP_max",
                                               feature_name = "nhiệt độ cực đại",
                                               display      = display,
                                               # ax           = axes[5,1]
                                               )
            
        # Option 3
        if "Nina_index" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "Nina_index",
                                               feature_name = "Nina_index",
                                               display      = display,
                                               # ax           = axes[0,0]
                                               )

        # Option 4
        if "DEW_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "DEW_ave",
                                               feature_name = "điểm sương trung bình",
                                               display      = display,
                                               # ax           = axes[0,1]
                                               )  
        # Option 5
        if "RH_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "RH_ave",
                                               feature_name = "độ ẩm tương đối trung bình",
                                               display      = display,
                                               # ax           = axes[1,1]
                                               )  
        # Option 6
        if "DEW_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "DEW_max",
                                               feature_name = "điểm sương cực đại",
                                               display      = display,
                                               # ax           = axes[2,0]
                                               )

        # Option 7
        if "RH_max" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "RH_max",
                                               feature_name = "độ ẩm tương đối cực đại",
                                               display      = display,
                                               # ax           = axes[2,1]
                                               )
        
        # Option 8
        if "sp_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "sp_ave",
                                               feature_name = "áp suất bề mặt trung bình",
                                               display      = display,
                                               # ax           = axes[3,0]
                                               )
            
        # Option 9
        if "tcc_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "tcc_ave",
                                               feature_name = "lượng mây che phủ trung bình",
                                               display      = display,
                                               # ax           = axes[3,1]
                                               )
        
        # Option 10
        if "tp_sum" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "tp_sum",
                                               feature_name = "lượng mưa tích lũy",
                                               display      = display,
                                               # ax           = axes[4,0]
                                               )
        
        # Option 11
        if "ws_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "ws_ave",
                                               feature_name = "tốc độ gió trung bình",
                                               display      = display,
                                               # ax           = axes[4,1]
                                               )
        
        # Option 12
        if "wd_ave" in features:
            plot_correlation_matrix_in_station(data         = df_filtered,
                                               data_cols    = "wd_ave",
                                               feature_name = "hướng gió trung bình",
                                               display      = display,
                                               # ax           = axes[5,0]
                                               )

def plot_trend_of_year_of_target(data, station, station_name,
                                 display = False,
                                 ax      = None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Lọc dữ liệu cho trạm Nội Bài
    station_data = data[data["NAME"] == station].copy()

    # Tính giá trị trung bình theo năm để giảm nhiễu
    yearly_data = station_data.groupby("YEAR").agg({"TEMP_ave": "mean",
                                                    "TEMP_max": "mean"
                                                    }).reset_index()

    # Tính giá trị trung bình toàn thời kỳ
    mean_Tave = yearly_data["TEMP_ave"].mean()
    mean_Tmax = yearly_data["TEMP_max"].mean()

    # Tính đường xu hướng tuyến tính
    z_Tave = np.polyfit(x   = yearly_data["YEAR"], 
                        y   = yearly_data["TEMP_ave"], 
                        deg = 1)
    p_Tave = np.poly1d(z_Tave)

    z_Tmax = np.polyfit(x   = yearly_data["YEAR"], 
                        y   = yearly_data["TEMP_max"], 
                        deg = 1)
    p_Tmax = np.poly1d(z_Tmax)

    if display is True:
        plt.figure(figsize=(14, 8))

        # Vẽ đường nhiệt độ và điểm dữ liệu
        plt.plot(yearly_data["YEAR"], 
                 yearly_data["TEMP_max"], 
                 marker = "o", 
                 color  = "steelblue", 
                 label  = "Trung bình nhiệt độ cực đại trong ngày (°C)")
        
        plt.plot(yearly_data["YEAR"], 
                 yearly_data["TEMP_ave"], 
                 marker = "o", 
                 color  = "darkgreen", 
                 label  = "Trung bình nhiệt độ trung bình trong ngày (°C)")

        # Vẽ đường xu hướng tuyến tính
        plt.plot(yearly_data["YEAR"], 
                 p_Tmax(yearly_data["YEAR"]), 
                 color     = "purple", 
                 linewidth = 2, 
                 label     = "Xu hướng tuyến tính")
        
        plt.plot(yearly_data["YEAR"], 
                 p_Tave(yearly_data["YEAR"]), 
                 color     = "purple", 
                 linewidth = 2)

        # Vẽ đường trung bình toàn thời kỳ
        plt.axhline(mean_Tmax, 
                    color     = "gray", 
                    linestyle = "--", 
                    label     = "Trung bình nhiệt độ 1990-2024")
        
        plt.axhline(mean_Tave, 
                    color     = "gray", 
                    linestyle = "--")

        plt.title(f"BIẾN ĐỘNG THEO NĂM CỦA ĐẶC TRƯNG NHIỆT ĐỘ \nTẠI TRẠM KHÍ TƯỢNG {station_name}", 
                    fontsize = 12, 
                    weight   = "bold")
        plt.xlabel("Năm")
        plt.ylabel("Nhiệt độ (°C)")
        plt.xticks(yearly_data["YEAR"][::2], rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.200), ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    return yearly_data

def plot_trend_of_year_of_features(data, station, station_name,
                                   display = False,
                                   ax      = None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Lọc dữ liệu cho trạm Nội Bài và tính trung bình theo năm
    station_data = data[data["NAME"] == station].copy()
    yearly_data = station_data.groupby("YEAR").agg({"Nina_index": "mean",
                                                    "DEW_ave": "mean",
                                                    "RH_ave": "mean", 
                                                    "DEW_max": "mean",
                                                    "RH_max": "mean",
                                                    "sp_ave": "mean",
                                                    "tcc_ave": "mean",
                                                    "tp_sum": "mean",
                                                    "ws_ave": "mean",
                                                    "wd_ave": "mean"
                                                    }).reset_index()

    if display is True:
        # Tạo subplot với nhiều trục y
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Trục trái: Nhiệt độ điểm sương và độ ẩm (°C và %)
        ax1.plot(yearly_data["YEAR"], 
                 yearly_data["DEW_ave"], 
                 color       = "red", 
                 marker      = "^", 
                 linewidth   = 2, 
                 label       = "Điểm sương trung bình", 
                 markersize  = 4)

        ax1.plot(yearly_data["YEAR"], 
                 yearly_data["DEW_max"], 
                 color      = "darkred", 
                 marker     = "v", 
                 linewidth  = 2, 
                 label      = "Điểm sương cực đại", 
                 markersize = 4)

        ax1.plot(yearly_data["YEAR"], 
                 yearly_data["RH_ave"], 
                 color      = "blue", 
                 marker     = "s", 
                 linewidth  = 2, 
                 label      = "Độ ẩm tương đối trung bình", 
                 markersize = 4)
        
        ax1.plot(yearly_data["YEAR"], 
                 yearly_data["RH_max"], 
                 color      = "darkblue", 
                 marker     = "D", 
                 linewidth  = 2, 
                 label      = "Độ ẩm tương đối cực đại", 
                 markersize = 4)

        ax1.set_xlabel("Năm", fontsize=12)
        ax1.set_ylabel("Điểm sương (°C) / Độ ẩm (%)", fontsize=12, color="black")
        ax1.tick_params(axis="y", labelcolor="black")

        # Trục phải 1: Áp suất và Nina index
        ax2 = ax1.twinx()
        ax2.plot(yearly_data["YEAR"], 
                 yearly_data["Nina_index"], 
                 color      = "purple", 
                 marker     = "o", 
                 linewidth  = 2, 
                 label      = "Nina index", 
                 markersize = 4)
        
        ax2.plot(yearly_data["YEAR"], 
                 yearly_data["sp_ave"], 
                 color      = "green", 
                 marker     = "*", 
                 linewidth  = 2, 
                 label      = "Áp suất bề mặt", 
                 markersize = 5)
        
        ax2.set_ylabel("Nina index / Áp suất (kPa)", fontsize=12, color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        # Trục phải 2: Các yếu tố khí tượng khác
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(yearly_data["YEAR"], 
                 yearly_data["tcc_ave"], 
                 color      = "gray", 
                 marker     = "h", 
                 linewidth  = 2, 
                 label      = "Độ che phủ mây", 
                 markersize = 4)

        ax3.plot(yearly_data["YEAR"], 
                 yearly_data["tp_sum"], 
                 color      = "cyan", 
                 marker     = "p", 
                 linewidth  = 2, 
                 label      = "Lượng mưa", 
                 markersize = 4)
        
        ax3.plot(yearly_data["YEAR"], 
                 yearly_data["ws_ave"], 
                 color      = "orange", 
                 marker     = "x", 
                 linewidth  = 2, 
                 label      = "Tốc độ gió", 
                 markersize = 5)

        ax3.set_ylabel("Mây (%) / Mưa (mm) / Gió (m/s)", fontsize=12, color="black")
        ax3.tick_params(axis="y", labelcolor="black")

        # Trục phải 3: Hướng gió
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.plot(yearly_data["YEAR"], 
                 yearly_data["wd_ave"], 
                 color      = "pink", 
                 marker     = "^", 
                 linewidth  = 2, 
                 label      = "Hướng gió trung bình", 
                 markersize = 4)
        
        ax4.set_ylabel("Hướng gió (độ)", fontsize=12, color="black")
        ax4.tick_params(axis="y", labelcolor="black")

        # Hiển thị năm cách 2 năm một lần và xoay 45 độ
        plt.xticks(yearly_data["YEAR"][::2], rotation=45)

        # Gộp legend từ tất cả các trục và đặt dưới xlabel
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax1.legend(lines1 
                   + lines2 
                   + lines3
                   + lines4
                   , 
                   labels1 
                   + labels2 
                   + labels3
                   + labels4
                   , 
                   loc='upper center', bbox_to_anchor=(0.5, -0.10), 
                   ncol=4, fontsize=10, frameon=False)

        # Tiêu đề
        plt.title( "BIẾN ĐỘNG CÁC ĐẶC TRƯNG ẢNH HƯỞNG ĐẾN NHIỆT ĐỘ CỰC ĐẠI \n"
                  f"TẠI TRẠM KHÍ TƯỢNG {station_name} (1990 - 2024)", 
                  fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.show()

    return yearly_data

def plot_trend_of_year_of_all_data(data, 
                                   method       = "full",
                                   stations     = list(["NOI BAI",
                                                        # "THANH HOA",
                                                        # "DONG HOI",
                                                        # "QUY NHON",
                                                        # "TSN",
                                                        # "CA MAU"
                                                        ]),
                                   display      = True,
                                   start_time   = None, 
                                   end_time     = None,
                                   freq         = None):
    import os
    import sys
    import pandas as pd    
    import matplotlib.pyplot as plt
    
    sys.path.append(os.path.abspath("../src"))    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")
        
    if method == "short":
        # Option 1
        if "NOI BAI" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "NOI BAI",
                                                              station_name = "NỘI BÀI",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "NOI BAI",
                                                                  station_name = "NỘI BÀI",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm NỘI BÀI 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")

        # Option 2
        if "THANH HOA" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "THANH HOA",
                                                              station_name = "THANH HÓA",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "THANH HOA",
                                                                  station_name = "THANH HÓA",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm THANH HÓA 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")

        # Option 3
        if "DONG HOI" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "DONG HOI",
                                                              station_name = "ĐỒNG HỚI",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "DONG HOI",
                                                                  station_name = "ĐỒNG HỚI",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm ĐỒNG HỚI 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")

        # Option 4
        if "QUY NHON" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "QUY NHON",
                                                              station_name = "QUY NHƠN",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "QUY NHON",
                                                                  station_name = "QUY NHƠN",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm QUY NHƠN 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")

        # Option 5
        if "TSN" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "TSN",
                                                              station_name = "TÂN SƠN NHẤT",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "TSN",
                                                                  station_name = "TÂN SƠN NHẤT",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm TÂN SƠN NHẤT 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")
            
        # Option 6
        if "CA MAU" in stations:
            yearly_data_target = plot_trend_of_year_of_target(data         = df_filtered,
                                                              station      = "CA MAU",
                                                              station_name = "CÀ MAU",
                                                              display      = False,
                                                              # ax           = None
                                                              )
            yearly_data_features = plot_trend_of_year_of_features(data         = df_filtered,
                                                                  station      = "CA MAU",
                                                                  station_name = "CÀ MAU",
                                                                  display      = False,
                                                                  # ax           = None
                                                                  )
            print("🔶🔶🔶 Trạm CÀ MAU 🔶🔶🔶")
            print(f"""Xu hướng biến đổi theo năm của nhiệt độ\n{yearly_data_target}
                                                             \n{yearly_data_target.describe()}\n""")
            print(f"""Xu hướng biến đổi theo năm của các đặc trưng\n{yearly_data_features}
                                                                  \n{yearly_data_features.describe()}\n""")

    elif method == "full":
        # fig, axes = plt.subplots(6, 2, figsize=(40, 80))
        # Option 1
        if "NOI BAI" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "NOI BAI",
                                         station_name = "NỘI BÀI",
                                         display      = display,
                                         # ax           = axes[1,0]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "NOI BAI",
                                           station_name = "NỘI BÀI",
                                           display      = display,
                                           # ax           = axes[1,0]
                                           )
            
        # Option 2
        if "THANH HOA" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "THANH HOA",
                                         station_name = "THANH HÓA",
                                         display      = display,
                                         # ax           = axes[5,1]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "THANH HOA",
                                           station_name = "THANH HÓA",
                                           display      = display,
                                           # ax           = axes[5,1]
                                           )
            
        # Option 3
        if "DONG HOI" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "DONG HOI",
                                         station_name = "ĐỒNG HỚI",
                                         display      = display,
                                         # ax           = axes[0,0]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "DONG HOI",
                                           station_name = "ĐỒNG HỚI",
                                           display      = display,
                                           # ax           = axes[0,0]
                                           )

        # Option 4
        if "QUY NHON" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "QUY NHON",
                                         station_name = "QUY NHƠN",
                                         display      = display,
                                         # ax           = axes[0,1]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "QUY NHON",
                                           station_name = "QUY NHƠN",
                                           display      = display,
                                           # ax           = axes[0,1]
                                           )   
             
        # Option 5
        if "TSN" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "TSN",
                                         station_name = "TÂN SƠN NHẤT",
                                         display      = display,
                                         # ax           = axes[1,1]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "TSN",
                                           station_name = "TÂN SƠN NHẤT",
                                           display      = display,
                                           # ax           = axes[1,1]
                                           )  
              
        # Option 6
        if "CA MAU" in stations:
            plot_trend_of_year_of_target(data         = df_filtered,
                                         station      = "CA MAU",
                                         station_name = "CÀ MAU",
                                         display      = display,
                                         # ax           = axes[2,0]
                                         )
            plot_trend_of_year_of_features(data         = df_filtered,
                                           station      = "CA MAU",
                                           station_name = "CÀ MAU",
                                           display      = display,
                                           # ax           = axes[2,0]
                                           )

def plot_trend_of_month_of_targets(data, station, station_name,
                                   display = False,
                                   ax      = None):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns    
    
    # Lấy dữ liệu trạm
    station_data = data[data["NAME"] == station].copy()
    
    # Resample theo tháng cho 2 target nhiệt độ
    monthly_features = station_data.resample("M").agg({"TEMP_ave": "mean",
                                                       "TEMP_max": "mean"})

    # Thêm cột năm và tháng
    monthly_features["Year"]  = monthly_features.index.year
    monthly_features["Month"] = monthly_features.index.month

    # Xác định giai đoạn
    def get_period(year):
        if 1990 <= year <= 2000:
            return "1990-2000"
        elif 2001 <= year <= 2010:
            return "2001-2010"
        elif 2011 <= year <= 2016:
            return "2011-2016"
        elif 2017 <= year <= 2024:
            return "2017-2024"
        else:
            return

    monthly_features["Period"] = monthly_features["Year"].apply(get_period)
    
    # Vẽ heatmap cho TEMP_ave (target)
    pivot_ave = monthly_features.groupby(["Period", "Month"])["TEMP_ave"].mean().unstack()
    # Thêm dòng tổng 1990-2024
    pivot_ave_all = pd.DataFrame(pivot_ave)
    pivot_ave_all.loc["1990-2024"] = monthly_features.groupby("Month")["TEMP_ave"].mean().values
    # Sắp xếp lại thứ tự giai đoạn
    order = ["1990-2024", "1990-2000", "2001-2010", "2011-2016", "2017-2024"]
    pivot_ave_all = pivot_ave_all.reindex(order)

    # Vẽ heatmap cho TEMP_max (target)
    pivot_max = monthly_features.groupby(["Period", "Month"])["TEMP_max"].mean().unstack()    
    # Thêm dòng tổng 1990-2024
    pivot_max_all = pd.DataFrame(pivot_max)
    pivot_max_all.loc["1990-2024"] = monthly_features.groupby("Month")["TEMP_max"].mean().values    
    # Sắp xếp lại thứ tự giai đoạn
    pivot_max_all = pivot_max_all.reindex(order)

    if display is True:
        fig, axes = plt.subplots(figsize=(14, 8))
        sns.heatmap(data     = pivot_ave_all, 
                    annot    = True,
                    square   = True, 
                    fmt      = ".1f", 
                    cmap     = "coolwarm", 
                    cbar_kws = {"label"  : "Nhiệt độ (°C)",
                                "shrink" : 0.5},
                    ax       = axes)
        
        axes.set_title(f"GIÁ TRỊ TRUNG BÌNH THÁNG\nCỦA NHIỆT ĐỘ TRUNG BÌNH TRONG NGÀY\nTẠI TRẠM KHÍ TƯỢNG {station_name}",
                         fontsize=14, fontweight='bold')
        axes.set_xlabel("Tháng")
        axes.set_ylabel("Giai đoạn")
        axes.tick_params(axis='y', rotation=0)
        
        fig, axes = plt.subplots(figsize=(14, 8))
        sns.heatmap(data     = pivot_max_all, 
                    annot    = True, 
                    square   = True, 
                    fmt      = ".1f", 
                    cmap     = "coolwarm", 
                    cbar_kws = {"label"  : "Nhiệt độ (°C)",
                                "shrink" : 0.5},
                    ax       = axes)

        axes.set_title(f"GIÁ TRỊ TRUNG BÌNH THÁNG\nCỦA NHIỆT ĐỘ CỰC ĐẠI TRONG NGÀY\nTẠI TRẠM KHÍ TƯỢNG {station_name}",
                         fontsize=14, fontweight='bold')
        axes.set_xlabel("Tháng")
        axes.set_ylabel("Giai đoạn")
        axes.tick_params(axis='y', rotation=0)
        
        os.makedirs("../edas/heatmap", exist_ok=True)
        plt.savefig(f"../edas/heatmap/trend_of_month_of_targets_{station}.png", dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return pivot_ave_all, pivot_max_all

def plot_trend_of_month_of_features(data, station, station_name,
                                    display = False,
                                    ax      = None):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns    
    
    # Lấy dữ liệu trạm
    station_data = data[data["NAME"] == station].copy()
    
    # Resample theo tháng cho features
    monthly_features = station_data.resample("M").agg({
        "DEW_ave": "mean",
        "DEW_max": "mean",
        "RH_ave": "mean",
        "RH_max": "mean",
        "Nina_index": "mean",
        "sp_ave": "mean",
        "tcc_ave": "mean",
        "tp_sum": "sum",
        "ws_ave": "mean",
        "wd_ave": "mean"
    })

    # Thêm cột năm và tháng
    monthly_features["Year"]  = monthly_features.index.year
    monthly_features["Month"] = monthly_features.index.month

    # Xác định giai đoạn
    def get_period(year):
        if 1990 <= year <= 2000:
            return "1990-2000"
        elif 2001 <= year <= 2010:
            return "2001-2010"
        elif 2011 <= year <= 2016:
            return "2011-2016"
        elif 2017 <= year <= 2024:
            return "2017-2024"
        else:
            return

    monthly_features["Period"] = monthly_features["Year"].apply(get_period)

    # Danh sách features và thông tin hiển thị
    features_info = {
        "DEW_ave": {"label": "Điểm sương trung bình", "unit": "°C"},
        "DEW_max": {"label": "Điểm sương cực đại", "unit": "°C"},
        "RH_ave": {"label": "Độ ẩm tương đối trung bình", "unit": "%"},
        "RH_max": {"label": "Độ ẩm tương đối cực đại", "unit": "%"},
        "Nina_index": {"label": "Nina index", "unit": ""},
        "sp_ave": {"label": "Áp suất bề mặt trung bình", "unit": "kPa"},
        "tcc_ave": {"label": "Độ che phủ mây trung bình", "unit": "%"},
        "tp_sum": {"label": "Lượng mưa tích lũy", "unit": "mm"},
        "ws_ave": {"label": "Tốc độ gió trung bình", "unit": "m/s"},
        "wd_ave": {"label": "Hướng gió trung bình", "unit": "°"}
    }
    
    periods = ["1990-2000", "2001-2010", "2011-2016", "2017-2024"]
    colors  = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    results = []
        
    if display is True:
        # Tạo subplot cho tất cả features
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        axes = axes.flatten()
        
    for idx, (feature, info) in enumerate(features_info.items()):
        if display is True:
            ax = axes[idx]
        
        for period, color, marker in zip(periods, colors, markers):
            period_data = monthly_features[monthly_features["Period"] == period]
            if not period_data.empty:
                monthly_avg = period_data.groupby("Month")[feature].mean()
                results.append(monthly_avg.rename(feature + "_" + period))
                if display is True:
                    ax.plot(monthly_avg.index, monthly_avg.values, 
                            marker=marker, color=color, linestyle='-', 
                            label=period, linewidth=2, markersize=6)
        if display is True:
            ax.set_xlabel("Tháng", fontsize=12)
            ax.set_ylabel(f"{info['label']} ({info['unit']})", fontsize=12)
            ax.set_title(f"Giá trị của {info['label']} tại trạm khí tượng {station_name}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks(range(1, 13))        
    if display is True:
        os.makedirs("../edas/lineplot", exist_ok=True)
        plt.savefig(f"../edas/lineplot/trend_of_month_of_features_{station}.png", dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)

def plot_trend_of_month_of_all_data(data, 
                                    method       = "full",
                                    stations     = list(["NOI BAI",
                                                         # "THANH HOA",
                                                         # "DONG HOI",
                                                         # "QUY NHON",
                                                         # "TSN",
                                                         # "CA MAU"
                                                         ]),
                                    display      = True,
                                    start_time   = None, 
                                    end_time     = None,
                                    freq         = None):
    import os
    import sys
    import pandas as pd    
    import matplotlib.pyplot as plt
    
    sys.path.append(os.path.abspath("../src"))    
    from src.utilities.dataset import HandleMissing_interpolate
    
    start_time = pd.to_datetime(start_time)
    end_time   = pd.to_datetime(end_time)
    df_filtered = data.copy()
    if start_time:
        df_filtered = df_filtered[df_filtered.index >= start_time]
    if end_time:
        df_filtered = df_filtered[df_filtered.index <= end_time]
    if freq:        
        # Chỉ giữ các cột số
        numeric_cols = df_filtered.select_dtypes(include='number').columns
        df_filtered = HandleMissing_interpolate(data   = df_filtered[numeric_cols].resample(freq).mean(),
                                                method = "time")
        
    if method == "short":            
        # Option 1
        if "NOI BAI" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "NOI BAI",
                                                                                          station_name = "NỘI BÀI",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "NOI BAI",
                                                                    station_name = "NỘI BÀI",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm NỘI BÀI 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")

        # Option 2
        if "THANH HOA" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "THANH HOA",
                                                                                          station_name = "THANH HÓA",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "THANH HOA",
                                                                    station_name = "THANH HÓA",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm THANH HÓA 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")
            
        # Option 3
        if "DONG HOI" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "DONG HOI",
                                                                                          station_name = "ĐỒNG HỚI",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "DONG HOI",
                                                                    station_name = "ĐỒNG HỚI",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm ĐỒNG HỚI 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")

        # Option 4
        if "QUY NHON" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "QUY NHON",
                                                                                          station_name = "QUY NHƠN",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "QUY NHON",
                                                                    station_name = "QUY NHƠN",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm QUY NHƠN 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")   
             
        # Option 5
        if "TSN" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "TSN",
                                                                                          station_name = "TÂN SƠN NHẤT",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "TSN",
                                                                    station_name = "TÂN SƠN NHẤT",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm TÂN SƠN NHẤT 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")  
              
        # Option 6
        if "CA MAU" in stations:
            monthly_data_temp_ave, monthly_data_temp_max = plot_trend_of_month_of_targets(data         = df_filtered,
                                                                                          station      = "CA MAU",
                                                                                          station_name = "CÀ MAU",
                                                                                          display      = False,
                                                                                          # ax           = None
                                                                                          )
            monthly_data_features = plot_trend_of_month_of_features(data         = df_filtered,
                                                                    station      = "CA MAU",
                                                                    station_name = "CÀ MAU",
                                                                    display      = False,
                                                                    # ax           = None
                                                                    )
            print("🔶🔶🔶 Trạm CÀ MAU 🔶🔶🔶")
            print(f"""Giá trị trung bình tháng của nhiệt độ trung bình\n{monthly_data_temp_ave}
                                                                      \n{monthly_data_temp_ave.describe()}\n""")
            print(f"""Giá trị trung bình tháng của nhiệt độ cực đại\n{monthly_data_temp_max}
                                                                      \n{monthly_data_temp_max.describe()}\n""")
            print(f"""Giá trị trung bình tháng của các đặc trưng\n{monthly_data_features}
                                                                \n{monthly_data_features.describe()}\n""")
         
    elif method == "full":        
        # fig, axes = plt.subplots(6, 2, figsize=(40, 80))            
        # Option 1
        if "NOI BAI" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "NOI BAI",
                                           station_name = "NỘI BÀI",
                                           display      = display,
                                           # ax           = axes[1,0]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "NOI BAI",
                                            station_name = "NỘI BÀI",
                                            display      = display,
                                            # ax           = axes[1,0]
                                            )
            
        # Option 2
        if "THANH HOA" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "THANH HOA",
                                           station_name = "THANH HÓA",
                                           display      = display,
                                           # ax           = axes[5,1]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "THANH HOA",
                                            station_name = "THANH HÓA",
                                            display      = display,
                                            # ax           = axes[5,1]
                                            )
            
        # Option 3
        if "DONG HOI" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "DONG HOI",
                                           station_name = "ĐỒNG HỚI",
                                           display      = display,
                                           # ax           = axes[0,0]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "DONG HOI",
                                            station_name = "ĐỒNG HỚI",
                                            display      = display,
                                            # ax           = axes[0,0]
                                            )

        # Option 4
        if "QUY NHON" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "QUY NHON",
                                           station_name = "QUY NHƠN",
                                           display      = display,
                                           # ax           = axes[0,1]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "QUY NHON",
                                            station_name = "QUY NHƠN",
                                            display      = display,
                                            # ax           = axes[0,1]
                                            )   
             
        # Option 5
        if "TSN" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "TSN",
                                           station_name = "TÂN SƠN NHẤT",
                                           display      = display,
                                           # ax           = axes[1,1]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "TSN",
                                            station_name = "TÂN SƠN NHẤT",
                                            display      = display,
                                            # ax           = axes[1,1]
                                            )  
              
        # Option 6
        if "CA MAU" in stations:
            plot_trend_of_month_of_targets(data         = df_filtered,
                                           station      = "CA MAU",
                                           station_name = "CÀ MAU",
                                           display      = display,
                                           # ax           = axes[2,0]
                                           )
            plot_trend_of_month_of_features(data         = df_filtered,
                                            station      = "CA MAU",
                                            station_name = "CÀ MAU",
                                            display      = display,
                                            # ax           = axes[2,0]
                                            )

def actual_vs_predict_line_plot(station, target, y_train_encoded, station_fit, scaler_y,
                                set_name = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf

    for name in station.keys():
        plt.figure(figsize=(15, 5))
        # true_fit
        sns.lineplot(x     = station_fit[name].index,
                     y     = scaler_y[name].inverse_transform(y_train_encoded[name][[target]]).ravel(),
                     color = 'blue',
                     label = 'Actual',
                     #  ax    = axes[i],
                     linewidth=1.5)
        # fit
        sns.lineplot(x     = station_fit[name].index,
                     y     = tf.squeeze(scaler_y[name].inverse_transform(station_fit[name])),
                     color = 'orange',
                     label = 'Predict',
                     #  ax    = axes[i],
                     linewidth=1.5,
                     alpha=0.8)
        plt.title(f"{set_name} Set: Actual vs Predict - {name}", fontsize=14, fontweight='bold')
        plt.grid()
        plt.tight_layout()
        plt.show()

def actual_vs_predict_scatter_plot(station, src_name, target, y_train_encoded, station_fit, scaler_y,
                                   set_name = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    import numpy as np

    num_cols = len(station_fit.keys())

    # Xác định số hàng và số cột hợp lý
    ncols = 2  # Số biểu đồ trên mỗi hàng
    nrows = int(np.ceil(num_cols / ncols))  # Tính số hàng cần thiết

    fig, ax = plt.subplots(ncols              = ncols, 
                           nrows              = nrows, 
                           figsize            = (5*ncols, 4*nrows),
                           constrained_layout = True)  # auto căn chỉnh
    ax = ax.flatten()  # chuyển mảng 2 chiều thành 1 chiều để dễ duyệt
    count = list(["a", "b", "c", "d", "e", "f"])
    for i, name in enumerate(station.keys(), 0):
        # fit
        sns.scatterplot(x  = scaler_y[name].inverse_transform(y_train_encoded[name][[target]]).ravel(),
                        y  = tf.squeeze(scaler_y[name].inverse_transform(station_fit[name])),
                        ax = ax[i])
        ax[i].set_xlabel("Thực tế")
        ax[i].set_ylabel("Dự đoán")
        # true_fit
        ax[i].plot(scaler_y[name].inverse_transform(y_train_encoded[name][[target]]),
                   scaler_y[name].inverse_transform(y_train_encoded[name][[target]]),
                   "r--")
        ax[i].set_title(f"({count[i]}) {src_name[name]}")
        plt.suptitle(f"""So sánh giá trị thực tế và giá trị dự đoán nhiệt độ cực đại\ntại các trạm trên tập {set_name}""", 
                     fontsize   = 15, 
                     fontweight = 'bold', 
                     ha         = 'center')
    plt.show()

def plot_metric_panel(stations, station_names, metric,
                      color   = None,
                      label   = None,
                      display = False,
                      ax      = None):
        import numpy as np
        
        x = np.arange(len(stations))
        if display is True:
            ax.plot(x, metric, marker='o', color=color, label=label, linewidth=1.8)
            ax.set_xticks(x)
            ax.set_xticklabels([station_names.get(s, s) for s in stations])
            ax.set_xlabel('Trạm khí tượng')            

            if label == 'R2':
                ax.set_ylabel('R2 (%)')
                ax.set_title(f'(a) Hệ số xác định R2')    
            else:
                ax.set_ylabel('Chỉ số lỗi')
                ax.set_title(f'(b) Chỉ số lỗi')
            ax.legend()
            ax.grid(axis='both', linestyle='--')

def plot_metrics_in_all_station(station, station_names, metrics_dict,
                                datasets     = dict({
                                                    # 'train': 'huấn luyện',
                                                    # 'valid': 'xác thực',
                                                    # 'test' : 'kiểm tra'
                                                    }),                        
                                metric_names = list([
                                                    #   'R2', 
                                                    #   'MAE', 
                                                    #   'MSE', 
                                                    #   'MSLE', 
                                                    #   'MAPE'
                                                    ]),
                                display  = True):
    import numpy as np
    import matplotlib.pyplot as plt

    for ds, info in datasets.items():
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        stations = list(station.keys())
        if "R2" in metric_names:
            plot_metric_panel(stations      = stations,
                              station_names = station_names,
                              metric        = [float(metrics_dict.get("R2", {}).get(ds, {}).get(s, np.nan) * 100.0) for s in stations],
                              color         = '#1f77b4',
                              label         = 'R2',
                              display       = display,
                              ax            = ax[0][0])

        if "MAE" in metric_names:
            plot_metric_panel(stations      = stations,
                              station_names = station_names,
                              metric        = [metrics_dict.get("MAE", {}).get(ds, {}).get(s, np.nan) for s in stations],
                              color         = "#d62728",
                              label         = 'MAE',
                              display       = display,
                              ax            = ax[0][1])
        if "MSE" in metric_names:
            plot_metric_panel(stations      = stations,
                              station_names = station_names,
                              metric        = [metrics_dict.get("MSE", {}).get(ds, {}).get(s, np.nan) for s in stations],
                              color         = '#1f77b4',
                              label         = 'MSE',
                              display       = display,
                              ax            = ax[0][1])

        if "MSLE" in metric_names:
            plot_metric_panel(stations      = stations,
                              station_names = station_names,
                              metric        = [metrics_dict.get("MSLE", {}).get(ds, {}).get(s, np.nan) for s in stations],
                              color         = '#ff7f0e',
                              label         = 'MSLE',
                              display       = display,
                              ax            = ax[1][0])

        if "MAPE" in metric_names:
            plot_metric_panel(stations      = stations,
                              station_names = station_names,
                              metric        = [metrics_dict.get("MAPE", {}).get(ds, {}).get(s, np.nan) for s in stations],
                              color         = '#2ca02c',
                              label         = 'MAPE',
                              display       = display,
                              ax            = ax[1][1])

            fig.suptitle(f'Các chỉ số đánh giá mô hình trên tập {info}', 
                         fontsize   = 15, 
                         fontweight = 'bold',
                         ha         = 'center')
            plt.tight_layout()
            plt.show()
            
                
def shap_summary_plot(station, src_name, shap_vals, features_to_plot, feature_col,
                      set_name,
                      col_width ,
                      row_height,
                      ncols     ):
    import numpy as np
    import matplotlib.pyplot as plt
    import shap

    # Tính layout
    num_cols   = len(station.keys())
    nrows      = int(np.ceil(num_cols / ncols))

    fig, ax = plt.subplots(nrows   = nrows, 
                           ncols   = ncols, 
                           figsize = (col_width * ncols, row_height * nrows), 
                           squeeze = False)
    ax = ax.flatten() # chuyển mảng 2 chiều thành 1 chiều để dễ duyệt

    for i, name in enumerate(station.keys()):
        plt.sca(ax[i])
        count = list(["a", "b", "c", "d", "e", "f"])
        shap.summary_plot(shap_vals[name],
                          features      = features_to_plot[name][feature_col],
                          feature_names = features_to_plot[name][feature_col].columns,
                          show          = False,
                          plot_size     = (col_width, row_height))

        ax[i].set_title(f"({count[i]}) {src_name[name]}")
        ax[i].tick_params(axis='both', which='both')

    plt.suptitle(f"SHAP Summary tại các trạm khí tượng trên tập {set_name}",
                 fontsize   = 15, 
                 fontweight = 'bold', 
                 ha         = 'center')
    plt.tight_layout()
    plt.show()

def comparasion_model(station, station_names, metrics_dict, 
                      datasets     = None,                        
                      metric_names = None,
                      ncols        = 2):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(station))
    x_labels = [station_names.get(s, s) for s in station]
    
    marker_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

    for ds, info in datasets.items():
        n_metrics = len(metric_names)
        nrows = int(np.ceil(n_metrics / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                 figsize=(9*ncols, 4*nrows),
                                 constrained_layout=True)
        axes = axes.flatten() if n_metrics > 1 else axes

        for idx, metric in enumerate(metric_names):
            for i, model in enumerate(metrics_dict.keys()):
                vals = list([metrics_dict[model].get(metric, {}).get(ds, {}).get(s, np.nan) * (100 if metric == 'R2' else 1)
                             for s in station])
                    
                axes[idx].plot(x, vals, marker=marker_list[i], label=model)
                axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
                axes[idx].set_xticks(x)
                axes[idx].set_xticklabels(x_labels)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3, linestyle='--')

        for j in range(n_metrics, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'So sánh độ chính xác dự báo của các mô hình trên tập {info}', fontsize=16, fontweight='bold', y=1.03)
        plt.show()