import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder  # Encode feature
from sklearn.preprocessing import OrdinalEncoder # Encode feature
from sklearn.preprocessing import MinMaxScaler   # Scale feature
from sklearn.preprocessing import LabelEncoder   # Encode target
# from scipy.stats import boxcox # Normalized feature

from sklearn.feature_selection import mutual_info_classif # PCA
from sklearn.model_selection   import train_test_split
from sklearn.model_selection   import RepeatedKFold
from sklearn.model_selection   import GridSearchCV
from sklearn.model_selection   import validation_curve
from sklearn.model_selection   import learning_curve


from sklearn.naive_bayes  import MultinomialNB
from sklearn.naive_bayes  import BernoulliNB
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC
from xgboost              import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# Missing data
def ProportionMissing_aproach1(data):
    # how many total missing values do we have?
    total_rows    = data.shape[0]
    total_missing = data.isnull().sum()

    # percent of data that is missing
    percent_missing = (total_missing/total_rows) * 100
    print(percent_missing.sort_values(),"%",sep="")
def ProportionMissing_aproach2(data):
    total_rows = data.shape[0]
    total_missing = data.isnull().sum()

    percent_missing = (total_missing / total_rows) * 100
    percent_missing = percent_missing.sort_values(ascending=True)

    padding = max(1, int(len(percent_missing) * 0.05))

    plt.figure(figsize=(10, 0.5 * len(percent_missing)))
    y_pos = np.arange(len(percent_missing))

    cmap = plt.cm.Blues
    norm_percent = percent_missing / 100
    colors = cmap(norm_percent)

    bars = plt.barh(y_pos, percent_missing, color=colors, edgecolor='black')

    for i, v in enumerate(percent_missing):
        plt.text(v + 0.5, i, f"{v:.2f}%", va='center', fontsize=9)

    plt.yticks(y_pos, percent_missing.index)
    # plt.xlabel("Percentage of Missing Data (%)")
    plt.title("Proportion of Missing Values by Column", fontsize=14, fontweight='bold', pad=20)

    # Các mốc tham chiếu mới
    mocs = [20, 40, 60, 80, 100]
    for p in mocs:
        plt.axvline(p, color='gray', linestyle='--', linewidth=0.8)

    # Vẽ text mốc phía trên
    y_top = len(percent_missing) - 1 + padding
    for p in mocs:
        plt.text(p, y_top + 0.12, f'{p}%', ha='center', va='bottom', fontsize=8, color='gray')

    # Xóa xticks mặc định
    plt.xticks([])

    # Vẽ text mốc phía dưới
    y_bottom = -padding
    for p in mocs:
        plt.text(p, y_bottom - 0.2, f'{p}%', ha='center', va='top', fontsize=8, color='gray')

    plt.ylim(-padding, len(percent_missing) - 1 + padding)
    plt.xlim(0, 108)
    plt.grid(visible=False, axis='x')

    # plt.tight_layout()
    plt.show()
def HandleMissing_aproach1(data):
    cols_with_null = []
    for col in data.columns:
        if data[col].isnull().any():
            cols_with_null.append(col)

    data = data.drop(columns=cols_with_null)
    return data
def HandleMissing_aproach2(data):
    for col in data.columns:
        if data[col].isnull().any():
            mode_value = data[col].mode()[0]
            data[col]  = data[col].fillna(mode_value)
    return data

# Duplicate data
def ProportionDuplicate_aproach1(data):
    duplicate_rows = data.duplicated()
    total_duplicates = duplicate_rows.sum()

    if total_duplicates == 0:
        print("Không có dòng trùng.")
        return

    print(f"Tổng số dòng trùng: {total_duplicates} / {len(data)} ({(total_duplicates / len(data)) * 100:.2f}%)\n")

    # max_len = max(len(col) for col in data.columns) + 1

    # for col in data.columns:
    #     num_duplicates_in_col = data.loc[duplicate_rows, col].duplicated().sum()
    #     percent_in_col = (num_duplicates_in_col / total_duplicates) * 100
    #     print(f"{col:<{max_len}}: {num_duplicates_in_col} trùng trong {total_duplicates} dòng ({percent_in_col:.2f}%)")
def ProportionDuplicate_aproach2(data):
    duplicate_rows = data.duplicated()
    total_rows = len(data)
    total_duplicates = duplicate_rows.sum()
    total_non_duplicates = total_rows - total_duplicates

    counts = [total_non_duplicates, total_duplicates]
    labels = ['False (Unique)', 'True (Duplicate)']
    colors = ['#4CAF50', '#FF5722']

    plt.figure(figsize=(8, 3))
    bars = plt.barh([0, 1], counts, color=colors, edgecolor='black')

    max_count = max(counts)

    for i, v in enumerate(counts):
        percent = (v / total_rows) * 100
        text = f"{percent:.2f}%"
        plt.text(v + max_count * 0.01, i, text, va='center', ha='left', color='black', fontsize=10)

    # Set xlim rộng hơn một chút để chứa phần trăm bên phải
    plt.xlim(0, max_count * 1.12)

    plt.yticks([0, 1], labels)
    plt.title("Duplicate Rows Proportion", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Rows")

    plt.grid(visible=False, axis='x')
    # plt.tight_layout()
    plt.show()
def HandleDuplicate_aproach1(data):
    data_with_dup_dropped = data.drop_duplicates()

    # original = data.shape[0]
    # after    = data_with_dup_dropped.shape[0]

    # print(f"Original Data          : {1:.2%}")
    # print(f"After remove duplicate : {after/original:.2%}")
    return data_with_dup_dropped

# Mismatch data
def ProportionMismatch_aproach1(data, match_type):
    max_len = max(len(col) for col in data.columns) + 1

    for col in data.columns:
        if col in match_type:
            total_rows = data[col].size
            valid_values = match_type[col]

            if valid_values == 'Numerical':
                print(f"{col:<{max_len}}: Not applicable (numerical)")
                continue

            # Tìm các giá trị mismatch
            mismatches = [value for value in data[col] if value not in valid_values]

            if mismatches:
                percent_mismatch = (len(mismatches) / total_rows) * 100
                unique_mismatches = sorted(list(set(mismatches)))
                print(f"{col:<{max_len}}: {percent_mismatch:5.2f}% ~ {unique_mismatches}")
            else:
                print(f"{col:<{max_len}}:  0.00%")
        else:
            print(f"{col:<{max_len}}: Not applicable (unchecked)")
def ProportionMismatch_aproach2(data, match_type):
    mismatch_percents = {}

    for col in data.columns:
        if col in match_type:
            expected_values = match_type[col]

            # Nếu Numerical thì bỏ qua
            if expected_values == 'Numerical':
                continue

            total_rows = data[col].size
            # Tìm giá trị mismatch
            mismatches = [value for value in data[col] if value not in expected_values]

            percent_mismatch = (len(mismatches) / total_rows) * 100
            mismatch_percents[col] = percent_mismatch

    if not mismatch_percents:
        print("Không có cột nào để kiểm tra mismatch!")
        return

    # Chuyển thành Series để dễ sort và plot
    percent_mismatch_series = pd.Series(mismatch_percents).sort_values(ascending=True)

    padding = max(1, int(len(percent_mismatch_series) * 0.05))
    plt.figure(figsize=(10, 0.5 * len(percent_mismatch_series)))
    y_pos = np.arange(len(percent_mismatch_series))

    cmap = plt.cm.Oranges
    norm_percent = percent_mismatch_series / 100
    colors = cmap(norm_percent)

    bars = plt.barh(y_pos, percent_mismatch_series, color=colors, edgecolor='black')

    for i, v in enumerate(percent_mismatch_series):
        plt.text(v + 0.5, i, f"{v:.2f}%", va='center', fontsize=9)

    plt.yticks(y_pos, percent_mismatch_series.index)
    plt.title("Proportion of Mismatched Values by Column", fontsize=14, fontweight='bold', pad=20)

    # Các mốc tham chiếu
    mocs = [20, 40, 60, 80, 100]
    for p in mocs:
        plt.axvline(p, color='gray', linestyle='--', linewidth=0.8)

    # Vẽ text mốc phía trên
    y_top = len(percent_mismatch_series) - 1 + padding
    for p in mocs:
        plt.text(p, y_top + 0.12, f'{p}%', ha='center', va='bottom', fontsize=8, color='gray')

    # Xóa xticks mặc định
    plt.xticks([])

    # Vẽ text mốc phía dưới
    y_bottom = -padding
    for p in mocs:
        plt.text(p, y_bottom - 0.2, f'{p}%', ha='center', va='top', fontsize=8, color='gray')

    plt.ylim(-padding, len(percent_mismatch_series) - 1 + padding)
    plt.xlim(0, 108)
    plt.grid(visible=False, axis='x')

    # plt.tight_layout()
    plt.show()
def HandleMismatch_aproach1(data, match_type):
    cols_to_drop = []
    for col in data.columns:
        if col in match_type:
            valid_values = match_type[col]

            if valid_values == 'Numerical':
                continue

            # Kiểm tra xem có giá trị nào không hợp lệ không
            has_mismatch = False
            for value in data[col]:
                if value not in valid_values:
                    has_mismatch = True
                    break

            if has_mismatch:
                cols_to_drop.append(col)

    # Xoá các cột có mismatch
    data = data.drop(columns=cols_to_drop)
    return data
def HandleMismatch_aproach2(data, match_type):
    for col in data.columns:
        if col in match_type:
            valid_values = match_type[col]

            if valid_values == 'Numerical':
                continue

            mode_value = data[col].mode()[0]

            # Tạo mask với những giá trị không hợp lệ
            mask = ~data[col].isin(valid_values)

            # Thay những giá trị sai thành mode
            data.loc[mask, col] = mode_value

    return data
def HandleMismatch_aproach3(data, match_type):
    for col in data.columns:
        if col in match_type:
            valid_values = match_type[col]

            # Nếu là Numerical thì bỏ qua
            if valid_values == 'Numerical':
                continue

            # Xác định các giá trị không hợp lệ
            mask = ~data[col].isin(valid_values)

            # Thay các giá trị không hợp lệ thành -1
            data.loc[mask, col] = -1

    return data

## Outlier data
def plot_Outlier(data, data_cols, target):
    """
    Với mỗi thuộc tính số trong data_cols:
    - Hiển thị histplot phân phối giá trị theo class
    - Hiển thị boxplot giá trị theo class
    Mỗi thuộc tính nằm trên một hàng.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_cols = len(data_cols)

    ncols = 2   # histplot và boxplot
    nrows = num_cols  # mỗi biến 1 hàng

    fig, axes = plt.subplots(
        nrows   = nrows, 
        ncols   = ncols, 
        figsize = (6*ncols, 4*nrows)
    )

    for i, column in enumerate(data_cols):
        # Histplot
        sns.histplot(
            data = data, 
            x    = column, 
            hue  = target, 
            kde  = True,
            ax   = axes[i, 0]
        )
        axes[i, 0].set_title(f'Histogram: {column} by {target}')
        axes[i, 0].grid(True)

        # Boxplot
        sns.boxplot(
            data = data, 
            x    = target,
            y    = column,
            ax   = axes[i, 1]
        )
        axes[i, 1].set_title(f'Boxplot: {column} by {target}')
        axes[i, 1].set_xlabel(target)
        axes[i, 1].set_ylabel(column)
        axes[i, 1].grid(True)

    plt.tight_layout()
    plt.show()
def find_outlier_zscore(data):
    for col in data:
        max = data[col].mean()
        min = data[col].std()
        # IQR = Q3 - Q1
        lower_bound = max - 3 * min
        upper_bound = max + 3 * min

        outlier_idx = np.where((data[col] < lower_bound) | (data[col] > upper_bound))[0]
        percent = (len(outlier_idx) / data.shape[0]) * 100
        print(f"{col:20}: {len(outlier_idx)} outliers ({percent:.2f}%)")
def remove_outliers_zscore(data, data_cols):
    """
    Remove outliers iteratively using the z-score (mean ± 3*std) method until no outliers are found.

    Parameters:
    data (pd.DataFrame): The input dataframe.
    data_cols (list): List of numeric columns to check for outliers.

    Returns:
    pd.DataFrame: Dataframe after removing outliers.
    """
    original_size = data.shape[0]

    while True:
        total_outliers = 0
        outlier_idx = set()

        for col in data_cols:
            mean = data[col].mean()
            std  = data[col].std()

            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_idx.update(idx)
            total_outliers += len(idx)

        if total_outliers == 0:
            break  # không còn outlier thì thoát

        data = data.drop(index=outlier_idx)

    after_size = data.shape[0]
    print(f"Original Data Size           : {original_size}")
    print(f"After Removing Outliers Size : {after_size}")
    print(f"Data Retained Percentage     : {after_size / original_size:.2%}")

    return data
def find_outlier_iqr(data):
    for col in data:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_idx = np.where((data[col] < lower_bound) | (data[col] > upper_bound))[0]
        percent = (len(outlier_idx) / data.shape[0]) * 100
        print(f"{col:20}: {len(outlier_idx)} outliers ({percent:.2f}%)")
def remove_outliers_iqr(data, data_cols):
    """
    Remove outliers iteratively using the IQR method until no outliers are found.

    Parameters:
    data (pd.DataFrame): The input dataframe.
    data_cols (list): List of numeric columns to check for outliers.

    Returns:
    pd.DataFrame: Dataframe after removing outliers.
    """
    original_size = data.shape[0]

    while True:
        total_outliers = 0
        outlier_idx = set()  # gom index outlier từ tất cả các cột

        for col in data_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_idx.update(idx)
            total_outliers += len(idx)

        if total_outliers == 0:
            break  # nếu không còn outlier thì dừng

        data = data.drop(index=outlier_idx)

    after_size = data.shape[0]
    print(f"Original Data Size           : {original_size}")
    print(f"After Removing Outliers Size : {after_size}")
    print(f"Data Retained Percentage     : {after_size / original_size:.2%}")

    return data
def find_outlier_percentile(data):
    for col in data:
        max = data[col].quantile(0.99)
        min = data[col].quantile(0.01)
        # IQR = Q3 - Q1
        lower_bound = min
        upper_bound = max

        outlier_idx = np.where((data[col] < lower_bound) | (data[col] > upper_bound))[0]
        percent = (len(outlier_idx) / data.shape[0]) * 100
        print(f"{col:20}: {len(outlier_idx)} outliers ({percent:.2f}%)")
def remove_outliers_percentile(data, data_cols):
    """
    Remove outliers iteratively using the Percentile method until no outliers are found.

    Parameters:
    data (pd.DataFrame): The input dataframe.
    data_cols (list): List of numeric columns to check for outliers.

    Returns:
    pd.DataFrame: Dataframe after removing outliers.
    """
    original_size = data.shape[0]

    while True:
        total_outliers = 0
        outlier_idx = set()  # dùng set để tránh trùng index
    
        for col in data_cols:
            lower_bound = data[col].quantile(0.01)
            upper_bound = data[col].quantile(0.99)
    
            # Tìm outlier và cộng vào set
            idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_idx.update(idx)
            total_outliers += len(idx)
    
        if total_outliers == 0:
            break  # không còn outlier thì thoát
        
        data = data.drop(index=outlier_idx)

    after_size = data.shape[0]
    print(f"Original Data Size           : {original_size}")
    print(f"After Removing Outliers Size : {after_size}")
    print(f"Data Retained Percentage     : {after_size / original_size:.2%}")

    return data

# Distribution data
def frequency_counts(data, data_cols):
    for i in data_cols:
        print(data[i].value_counts().apply(
            lambda x: f"{x}/{len(data)}"
        ))
        print()  # cách 1 dòng trống
def frequency_distributions(data, data_cols):
    for i in data_cols:
        print(data[i].value_counts().apply(
           lambda x: f"{(x/len(data))*100:f}%"
        ))
        print()  # cách 1 dòng trống

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

# Best model estimator
def FindBestTuningModel(model, param_grid, train_X, train_y):
    cv = RepeatedKFold(
        n_splits  = 10,
        n_repeats = 10,
        random_state = 40020409
    )
    grid_cv = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = "accuracy",
        n_jobs = 5, # Import, as high as value ~ as much as precessore in order to run
        cv = cv,
        refit = True
    )
    grid_cv.fit(train_X,train_y)
    print(f"{type(model).__name__}'s best parameters: {grid_cv.best_params_}")

    cv_results   = pd.DataFrame(grid_cv.cv_results_).filter(like = "split")
    best_results = cv_results.loc[grid_cv.best_index_, :]

    print(f"Mean accuracy          : {best_results.mean()}")
    print(f"Accuracy std deviation : {best_results.std()}")

    return grid_cv.best_estimator_

# Model tuning
def plot_VC(X_data, Y_data, model, param_grid, n_jobs, ylim=None, log=True):
    for param_name, param_range in param_grid.items():
        train_scores, test_scores = validation_curve(
            model, X_data, Y_data,
            param_name  = param_name,
            param_range = param_range,
            scoring     = "accuracy",
            n_jobs      = n_jobs
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std  = np.std(train_scores, axis=1)
        test_scores_mean  = np.mean(test_scores, axis=1)
        test_scores_std   = np.std(test_scores, axis=1)

        # plt.figure(figsize=(8, 5))
        plt.title(f"Validation Curve for {param_name}")

        if ylim is not None:
            plt.ylim(*ylim)

        best_index       = np.argmax(test_scores_mean)
        best_param_value = param_range[best_index]
        best_score       = test_scores_mean[best_index]

        # 👉 Vẽ đường thẳng xanh nét đứt trước tiên
        plt.axvline(x=best_param_value, color='b', linestyle='--', label=f"Best Score = {best_score:.2f}")

        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        if log:
            plt.semilogx(param_range, train_scores_mean, 'o-', color="r", label="Training score")
            plt.semilogx(param_range, test_scores_mean, 'o-', color="g", label="Test score")
        else:
            plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Test score")

        y_text = best_score + 0.02 if ylim is None else min(best_score + 0.02, ylim[1] - 0.01)
        plt.text(best_param_value, y_text, f" Score: {best_score:.4f}\n {param_name}: {best_param_value}", 
                 color='b', fontsize=9, ha='left')

        plt.xlabel(f'Parameter: {param_name}')
        plt.ylabel('Score')
        plt.legend(loc="lower right")
        plt.grid()
        # plt.tight_layout()
        plt.show()

        # In kết quả ra ngoài terminal để tiện copy-paste
        print(f"Validation Curve for {param_name}:")
        print(f"  → Best {param_name} = {best_param_value}")
        print(f"  → Best CV Score     = {best_score:.4f}\n")
def plot_LC(X_data, Y_data, model, train_sizes, random_state, n_jobs):
    # plt.figure(figsize=(8, 5))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_data, Y_data, n_jobs=n_jobs, train_sizes=train_sizes, random_state=random_state
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    # === Tìm vị trí có test score cao nhất
    best_index = np.argmax(test_scores_mean)
    best_train_size = train_sizes[best_index]
    best_score = test_scores_mean[best_index]

    # Tính tỉ lệ train size đó trên tổng dữ liệu
    best_train_ratio = best_train_size / len(X_data)

    # Vẽ đường thẳng nét đứt tại điểm đó (vẽ trước để không đè lên line)
    plt.axvline(x=best_train_size, color='b', linestyle='--', label=f"Best CV Score = {best_score:.2f}")

    # Vẽ vùng sai số
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean  - test_scores_std,  test_scores_mean  + test_scores_std,  alpha=0.1, color="g")

    # Vẽ đường train và test score
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean,  'o-', color="g", label="Test score")

    # Chú thích giá trị điểm số và train size tại vị trí đó
    plt.text(best_train_size + (train_sizes.max() * 0.02), best_score,
             f"Score: {best_score:.4f}\nTrain size: {int(best_train_size)}\nRatio: {best_train_ratio:.4f}",
             color='b', fontsize=9, ha='left')

    plt.legend(loc="best")
    # plt.tight_layout()
    plt.show()

    # In kết quả ra ngoài terminal để tiện copy-paste
    print("Learning Curve Results:")
    print(f"  → Best train size   = {best_train_size} samples")
    print(f"  → Ratio             = {best_train_ratio:.4f}")
    print(f"  → Best CV Score     = {best_score:.4f}\n")
def plot_LC_multi_models(X_datasets, Y_data, models, model_names, train_sizes, random_state, n_jobs):
    plt.figure(figsize=(12, 7))
    plt.title("Learning Curves for Multiple Models")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Danh sách màu khác nhau
    colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])

    # Lưu trung bình test score của từng model để tính vị trí tốt nhất
    all_test_scores_mean = []

    for X_data, model, name in zip(X_datasets, models, model_names):
        color = next(colors)

        train_sizes_res, train_scores, test_scores = learning_curve(
            model, X_data, Y_data, n_jobs=n_jobs, train_sizes=train_sizes, random_state=random_state
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std  = np.std(train_scores, axis=1)
        test_scores_mean  = np.mean(test_scores, axis=1)
        test_scores_std   = np.std(test_scores, axis=1)

        all_test_scores_mean.append(test_scores_mean)

        # Đường train score - nét liền
        plt.plot(train_sizes_res, train_scores_mean, 'o-', color=color, label=f"{name} Train Score")
        plt.fill_between(train_sizes_res,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color=color)

        # Đường CV score - nét đứt
        plt.plot(train_sizes_res, test_scores_mean, 'o--', color=color, label=f"{name} CV Score")
        plt.fill_between(train_sizes_res,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color=color)

    # Tính trung bình test score tất cả models
    avg_test_scores = np.mean(all_test_scores_mean, axis=0)

    # Tìm train size tốt nhất
    best_index = np.argmax(avg_test_scores)
    best_train_size = train_sizes_res[best_index]
    best_score = avg_test_scores[best_index]

    # Tính tỷ lệ train_size / max_train_size
    train_size_ratio = best_train_size / max(train_sizes_res)

    # Vẽ đường thẳng tại train size tốt nhất
    plt.axvline(x=best_train_size, color='blue', linestyle='--', linewidth=1.5,
                label=f"Best avg CV Score = {best_score:.2f} at {best_train_size} ({train_size_ratio:.2%})")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print(f"\n=> Best average CV Score = {best_score:.4f} at train size = {best_train_size} ({train_size_ratio:.4f})")

# Model evaluation
def plot_CF_aproach1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
def plot_CF_aproach2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Tính tổng số mẫu
    total = np.sum(cm)

    # Tạo labels chứa số lượng và %
    labels = np.array([["{0}\n({1:.1f}%)".format(value, (value/total)*100)
                        for value in row] for row in cm])

    # Tạo DataFrame cho confusion matrix
    cm_matrix = pd.DataFrame(
        data    = cm, 
        columns = ['Actual Positive:1' , 'Actual Negative:0'], 
        index   = ['Predict Positive:1', 'Predict Negative:0']
    )

    # Vẽ heatmap với annot là labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        data  = cm_matrix, 
        annot = labels, 
        fmt   = '', 
        cmap  = 'YlGnBu'
    )
    plt.title('Confusion Matrix with Percentage')
    plt.show()