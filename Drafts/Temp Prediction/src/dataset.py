def grib_to_csv(src, des):
    import pandas as pd
    import cfgrib

    with cfgrib.open_dataset(src) as ds:
        df = ds.to_dataframe()
    df.to_csv(des)

def merge_df(datasets, on, how):
    import pandas as pd
    
    df_merged = None
    for df in datasets:
        if df_merged is None:
            df_merged = df
        else:
            df_merged = df_merged.merge(df, on=on, how=how)
    return df_merged
    
def compare_df(df_A, df_B, key):
    import pandas as pd
    # Merge lại
    merged = pd.merge(
        df_A[key],
        df_B[key],
        how="outer",
        indicator=True
    )

    # Lọc dòng khác nhau
    diff_rows = merged[merged["_merge"] != "both"]
    return diff_rows

def find_time_gaps(data, start_time, end_time, freq="1H"):
    import pandas as pd

    """
    Tìm các khoảng thời gian bị thiếu hoặc không đúng freq giữa start_time và end_time trong DataFrame
    Args:
        data (DataFrame): dữ liệu có DatetimeIndex
        start_time (str or Timestamp): thời gian bắt đầu
        end_time (str or Timestamp): thời gian kết thúc
        freq (str): tần suất mong đợi ('1H', '30min', ...)
    Returns:
        DataFrame với các cột: from_time, to_time, gap_duration
    """

    # Kiểm tra DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Convert timezone cho data.index về cùng tz với start_time nếu có
    if data.index.tz is None and pd.Timestamp(start_time).tz is not None:
        data.index = data.index.tz_localize(pd.Timestamp(start_time).tz)
    elif data.index.tz is not None and pd.Timestamp(start_time).tz is None:
        data.index = data.index.tz_convert(None)

    # Tạo time range chuẩn
    expected_times = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Tìm missing times
    missing_times = expected_times.difference(data.index)

    # Nếu không thiếu thì trả về luôn
    if missing_times.empty:
        print("✅ Không có time gap.")
        return None

    # Tính các đoạn gap liên tiếp
    gaps = []
    current_gap = [missing_times[0]]

    for t in missing_times[1:]:
        if t - current_gap[-1] == pd.Timedelta(freq):
            current_gap.append(t)
        else:
            gaps.append((current_gap[0], current_gap[-1], current_gap[-1] - current_gap[0] + pd.Timedelta(freq)))
            current_gap = [t]

    # Thêm đoạn cuối
    if current_gap:
        gaps.append((current_gap[0], current_gap[-1], current_gap[-1] - current_gap[0] + pd.Timedelta(freq)))

    # Trả về DataFrame
    gap_df = pd.DataFrame(gaps, columns=["from_time", "to_time", "gap_duration"])

    return gap_df
def fill_time_gaps(data, start_time, end_time, freq="1H"):
    import pandas as pd

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    tzinfo = data.index.tz

    # Tạo time range cần fill
    expected_times = pd.date_range(start=start_time, end=end_time, freq=freq, tz=tzinfo)

    # Chỉ lấy data trong khoảng start_time và end_time
    data_in_range = data[(data.index >= start_time) & (data.index <= end_time)]

    # Reindex data trong khoảng đó
    data_filled = data_in_range.reindex(expected_times)

    # Gộp lại với dữ liệu ngoài khoảng (giữ nguyên)
    data_outside_range = data[(data.index < start_time) | (data.index > end_time)]

    final_df = pd.concat([data_outside_range, data_filled]).sort_index()

    # Thêm lại cột time
    final_df = final_df.reset_index().rename(columns={"index": "time"})

    return final_df


# Missing data
def ProportionMissing_aproach1(data):
    # how many total missing values do we have?
    total_rows    = data.shape[0]
    total_missing = data.isnull().sum()

    # percent of data that is missing
    percent_missing = (total_missing/total_rows) * 100
    print(percent_missing.sort_values(),"%",sep="")
def ProportionMissing_aproach2(data):
    from matplotlib import pyplot as plt
    import numpy as np
    
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
def HandleMissing_drop(data):
    cols_with_null = []
    for col in data.columns:
        if data[col].isnull().any():
            cols_with_null.append(col)

    data = data.drop(columns=cols_with_null)
    return data
def HandleMissing_fillna(data, method):
    import pandas as pd
    import numpy as np

    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                # Dữ liệu dạng categorical: fill bằng mode
                value = data[col].mode()[0]
                data[col] = data[col].fillna(value)
            else:
                # Dữ liệu số
                if method == "mode":
                    value = data[col].mode()[0]
                    data[col] = data[col].fillna(value)
                elif method == "mean":
                    value = data[col].mean()
                    data[col] = data[col].fillna(value)
                elif method == "median":
                    value = data[col].median()
                    data[col] = data[col].fillna(value)
                elif method == "ffill":
                    data[col] = data[col].ffill()
                elif method == "bfill":
                    data[col] = data[col].bfill()
                else:
                    raise ValueError("Method phải là 'mode', 'mean', 'median', 'ffill' hoặc 'bfill'")
    return data
def HandleMissing_interpolate(data, method):
    import pandas as pd

    # Lưu lại timezone hiện có nếu có
    tzinfo = data.index.tz if isinstance(data.index, pd.DatetimeIndex) else None

    # Nếu có timezone, tạm bỏ để interpolate
    if tzinfo is not None:
        data.index = data.index.tz_localize(None)

    for col in data.columns:
        if data[col].isnull().any():
            data[col] = data[col].interpolate(method=method)

    # Gán lại timezone cũ cho index
    if tzinfo is not None:
        data.index = data.index.tz_localize(tzinfo)

    return data


# Duplicate data
def ProportionDuplicate_aproach1(data):
    duplicate_rows = data.duplicated()
    total_duplicates = duplicate_rows.sum()

    if total_duplicates == 0:
        print("Không có dòng trùng.")
        return
    
    print(data[duplicate_rows])
    print(f"Tổng số dòng trùng: {total_duplicates} / {len(data)} ({(total_duplicates / len(data)) * 100:.2f}%)\n")

    # max_len = max(len(col) for col in data.columns) + 1

    # for col in data.columns:
    #     num_duplicates_in_col = data.loc[duplicate_rows, col].duplicated().sum()
    #     percent_in_col = (num_duplicates_in_col / total_duplicates) * 100
    #     print(f"{col:<{max_len}}: {num_duplicates_in_col} trùng trong {total_duplicates} dòng ({percent_in_col:.2f}%)")
def ProportionDuplicate_aproach2(data):
    from matplotlib import pyplot as plt

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
def HandleDuplicate_drop(data, subset, keep):
    data_with_dup_dropped = data.drop_duplicates(subset = subset,
                                                 keep   = keep)

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
    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    
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
def find_outlier(data, method):
    import numpy as np
    import pandas as pd

    # Chỉ lấy các cột số
    num_cols = data.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if method == "zscore":
            mean = data[col].mean()
            std  = data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        elif method == "iqr":
            Q1  = data[col].quantile(0.25)
            Q3  = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

        elif method == "percentile":
            lower_bound = data[col].quantile(0.01)
            upper_bound = data[col].quantile(0.99)

        else:
            raise ValueError("Method phải là 'zscore', 'iqr' hoặc 'percentile'")

        # Tìm index outlier
        outlier_idx = np.where((data[col] < lower_bound) | (data[col] > upper_bound))[0]
        percent = (len(outlier_idx) / data.shape[0]) * 100
        print(f"{col:20}: {len(outlier_idx)} outliers ({percent:.2f}%)")
def remove_outliers(data, method="zscore"):
    """
    Remove outliers in numerical columns using Z-score, IQR hoặc Percentile.

    Parameters:
    data (pd.DataFrame): The input dataframe.
    method (str): 'zscore', 'iqr' hoặc 'percentile'

    Returns:
    pd.DataFrame: Dataframe after removing outliers.
    """
    import numpy as np
    import pandas as pd

    original_size = data.shape[0]
    outlier_idx = set()

    # Chỉ lấy các cột numeric
    numeric_cols = data.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if method == "zscore":
            mean = data[col].mean()
            std  = data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        elif method == "iqr":
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

        elif method == "percentile":
            lower_bound = data[col].quantile(0.01)
            upper_bound = data[col].quantile(0.99)

        else:
            raise ValueError("Method phải là 'zscore', 'iqr' hoặc 'percentile'")

        idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outlier_idx.update(idx)

    data_cleaned = data.drop(index=outlier_idx)

    after_size = data_cleaned.shape[0]
    print(f"Original Data Size           : {original_size}")
    print(f"After Removing Outliers Size : {after_size}")
    print(f"Data Retained Percentage     : {after_size / original_size:.2%}")

    return data_cleaned
def winsorize_outliers(data, 
                       data_cols  = None, 
                       method     = "zscore", 
                       z_thresh   = 3.0, 
                       iqr_factor = 1.5, 
                       lower_pct  = 0.01, 
                       upper_pct  = 0.99, 
                       verbose    = True):
    """
    Capping outliers trong numerical columns.

    Parameters:
        data (pd.DataFrame): Dữ liệu gốc.
        method (str): 'zscore', 'iqr', hoặc 'percentile'
        data_cols (list): Danh sách cột cần xử lý. Mặc định tất cả numeric columns.
        z_thresh (float): Ngưỡng z-score nếu method='zscore'
        iqr_factor (float): Hệ số nhân IQR nếu method='iqr'
        lower_pct (float): Percentile thấp nếu method='percentile'
        upper_pct (float): Percentile cao nếu method='percentile'
        verbose (bool): In log xử lý.

    Returns:
        pd.DataFrame: Dữ liệu sau khi capping outlier.
    """
    data_capped = data.copy()
    if data_cols is None:
        data_cols = data.select_dtypes(include=["number"]).columns

    for col in data_cols:
        outlier_count = 0
        if method == "zscore":
            mean = data[col].mean()
            std  = data[col].std()
            lower, upper = mean - z_thresh * std, mean + z_thresh * std

        elif method == "iqr":
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR

        elif method == "percentile":
            lower, upper = data[col].quantile(lower_pct), data[col].quantile(upper_pct)

        else:
            raise ValueError("Phương pháp phải là 'zscore', 'iqr', hoặc 'percentile'.")

        mask_lower = data_capped[col] < lower
        mask_upper = data_capped[col] > upper
        outlier_count = mask_lower.sum() + mask_upper.sum()

        data_capped.loc[mask_lower, col] = lower
        data_capped.loc[mask_upper, col] = upper

        if verbose:
            print(f"📊 {col}: {outlier_count} outlier(s) capped bằng {method}")

    if verbose:
        print(f"✅ Đã xử lý outliers bằng phương pháp: {method}\n")

    return data_capped
def transform_outliers(data,
                       data_cols = None, 
                       method    = "log", 
                       style     = "replace", 
                       verbose   = True):
    from scipy import stats
    
    """
    Transform numerical feature để giảm ảnh hưởng outlier.

    Parameters:
        data (DataFrame): Dữ liệu gốc.
        method (str): 'log', 'sqrt', 'boxcox', 'yeojohnson'
        data_cols (list): Danh sách cột cần xử lý. Mặc định tất cả numeric columns.
        style (str): 'replace' hoặc 'create_new'
        verbose (bool): In log xử lý.

    Returns:
        DataFrame: Dữ liệu sau khi transform
    """
    data_transformed = data.copy()
    if data_cols is None:
        data_cols = data.select_dtypes(include=["number"]).columns

    for col in data_cols:
        try:
            if method == "log":
                if (data[col] < 0).any():
                    raise ValueError("Log transform yêu cầu giá trị ≥ 0.")
                transformed = np.log1p(data[col])

            elif method == "sqrt":
                transformed = np.sqrt(data[col].clip(lower=0))

            elif method == "boxcox":
                transformed, _ = stats.boxcox(data[col].dropna() + 1)
                transformed = pd.Series(transformed, index=data[col].dropna().index)

            elif method == "yeojohnson":
                transformed, _ = stats.yeojohnson(data[col].dropna())
                transformed = pd.Series(transformed, index=data[col].dropna().index)

            else:
                raise ValueError("Phương pháp phải là 'log', 'sqrt', 'boxcox' hoặc 'yeojohnson'.")

            if style == "replace":
                data_transformed.loc[transformed.index, col] = transformed

            elif style == "create_new":
                new_col = f"{col}_{method}"
                data_transformed.loc[transformed.index, new_col] = transformed

            else:
                raise ValueError("Style phải là 'replace' hoặc 'create_new'.")

            if verbose:
                print(f"🔄 {col} transform thành công bằng {method}")

        except Exception as e:
            if verbose:
                print(f"⚠️ Lỗi xử lý {col}: {e}")

    return data_transformed

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

# LSTM autoencoder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras import Model
import os
class LSTMAutoencoder(Model):    
    def __init__(self):
        super(LSTMAutoencoder, self).__init__()
        self.model = None
        self.history = None
        self.train_mae_loss = None
        self.test_mae_loss = None
        self.anomalies = None
        self.score_df = None

    def create_dataset(self, X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    def prepare_data(self, train_data, test_data, target_column, time_steps):
        """
        Chuẩn bị dữ liệu cho training và testing
        """        
        self.target_column = target_column
        self.train_data = train_data
        self.test_data = test_data
        self.time_steps = time_steps

        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(train_data[[target_column]])

        # Scale dữ liệu
        train_data[target_column] = self.scaler.transform(train_data[[target_column]])
        test_data[target_column] = self.scaler.transform(test_data[[target_column]])
        
        self.X_train, self.y_train = self.create_dataset(train_data[[target_column]], train_data[target_column], time_steps)
        self.X_test, self.y_test = self.create_dataset(test_data[[target_column]], test_data[target_column], time_steps)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_autoencoder(self):
        """
        Xây dựng model model
        """
        LSTM_units = 64
        model = keras.Sequential([
            layers.LSTM(LSTM_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=False,name='encoder_lstm'),
            layers.Dropout(0.2, name='encoder_dropout'),
            layers.RepeatVector(self.X_train.shape[1], name='decoder_repeater'),
            layers.LSTM(LSTM_units, return_sequences=True, name='decoder_lstm'),
            layers.Dropout(0.2, name='decoder_dropout'),
            layers.TimeDistributed(layers.Dense(self.X_train.shape[2]),name='decoder_dense_output')
        ])

        model.compile(optimizer='adam', loss='mae')
        self.model = model
        return model
    
    def train(self, epochs=20, batch_size=256, validation_split=0.1, patience=5, model_path=None):
        """
        Training model
        """
        if self.model is None:
            self.build_autoencoder()
            
        cp = tf.keras.callbacks.ModelCheckpoint(
            f'../models/LSTMAutoencoders/trained_model/{model_path}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        es = tf.keras.callbacks.EarlyStopping(
            restore_best_weights=True, 
            patience=patience
        )
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[es, cp],
            shuffle=False
        )

    def plot_training_history(self):
        """
        Vẽ biểu đồ training history
        """
        if self.history is None:
            print("Model chưa được train!")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()

    def evaluate(self):
        """
        Đánh giá model
        """
        if self.model is None:
            print("Model chưa được train!")
            return
            
        test_loss = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {test_loss}")
        return test_loss
    

    def detect_anomalies(self, full_data, method='std'):
        """
        Áp dụng mô hình đã train để detect và reconstruct anomalies trên toàn bộ chuỗi thời gian
        """
        if self.model is None:
            print("Bạn cần train model và tính toán anomaly scores trước!")
            return       

        # Scale full data
        self.scaled_full = full_data.copy()
        self.scaled_full[self.target_column] = self.scaler.transform(full_data[[self.target_column]])

        # Tạo tập dữ liệu sequence
        X_full, y_full = self.create_dataset(self.scaled_full[[self.target_column]], self.scaled_full[self.target_column], self.time_steps)

        # Dự đoán tái tạo
        self.X_pred = self.model.predict(X_full)
        mae_loss = np.mean(np.abs(self.X_pred - X_full), axis=1)
            
        # Tính threshold từ training
        if method == 'std':
            threshold = np.mean(mae_loss) + 3 * np.std(mae_loss)

        # elif method == 'quantile':
        #     threshold = np.quantile(mae_loss, 0.97)

        # elif method == 'gmm':
        #     from sklearn.mixture import GaussianMixture
        #     gmm = GaussianMixture(n_components=2, random_state=0)
        #     gmm.fit(mae_loss)
        #     # Lấy cluster có mean lớn hơn làm "anomaly"
        #     means = gmm.means_.flatten()
        #     anomaly_cluster = np.argmax(means)
        #     scores = gmm.predict_proba(mae_loss)[:, anomaly_cluster]
        #     threshold = np.percentile(mae_loss[scores > 0.5], 5)  # rất nhạy cảm
        
        # Gắn thông tin anomaly vào dataframe
        self.score_df = full_data[self.time_steps:].copy().reset_index(drop=True)
        self.score_df['loss'] = mae_loss
        self.score_df['threshold'] = threshold
        self.score_df['anomaly'] = self.score_df.loss > self.score_df.threshold
        anomalies = self.score_df[self.score_df.anomaly == True]
        
        print(f"Tổng số anomalies phát hiện: {len(anomalies)}")
        print(f"Tỷ lệ anomalies: {len(anomalies)/len(self.score_df)*100:.2f}%")

        return anomalies
    
    def plot_anomaly_scores(self, full_data, time_column='time'):
        """
        Vẽ biểu đồ anomaly scores
        """
        if self.score_df is None:
            print("Chưa phát hiện anomalies!")
            return
        
        plot_df = self.score_df.copy()
        plot_df['time'] = full_data[self.time_steps:][time_column].values
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=plot_df, x='time', y='loss', label='Test Loss')
        sns.lineplot(data=plot_df, x='time', y='threshold', label='Threshold')
        plt.title('Anomaly Scores')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def reconstruct_and_replace_anomalies(self, inverse=True):
        # Reconstruct anomalies
        corrected = self.scaled_full[self.time_steps:][[self.target_column]].copy().reset_index(drop=True)
        anomaly_indices = self.score_df[self.score_df['anomaly']].index

        for idx in anomaly_indices:
            corrected_value = self.X_pred[idx, -1, 0]
            corrected.at[idx, self.target_column] = corrected_value

        if inverse:
            corrected[self.target_column] = self.scaler.inverse_transform(corrected[[self.target_column]])

        # Gắn corrected vào score_df để tiện phân tích
        self.score_df['corrected'] = corrected[self.target_column]
        return self.score_df


    def plot_reconstruct_result(self, time_column='time'):
        """
        Vẽ 2 biểu đồ: dữ liệu gốc với anomaly và dữ liệu đã được corrected, dùng cùng scale trục y
        """
        if 'corrected' not in self.score_df.columns:
            print("Thiếu dữ liệu corrected! Hãy chắc chắn đã chạy detect_and_reconstruct_full_series.")
            return

        time_series = self.score_df[time_column].values
        original = self.score_df[self.target_column].values
        corrected = self.score_df['corrected'].values
        anomalies = self.score_df[self.score_df['anomaly']]

        # Xác định min/max để đồng nhất scale
        global_min = min(original.min(), corrected.min())
        global_max = max(original.max(), corrected.max())

        # Vẽ 2 biểu đồ nằm ngang
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

        # Biểu đồ dữ liệu gốc với anomaly
        axes[0].plot(time_series, original, label='Original', zorder=1)
        axes[0].scatter(anomalies[time_column].values,
                        anomalies[self.target_column].values,
                        color='red', marker='X', s=60, label='Anomaly', zorder=2)
        axes[0].set_title('Original Series with Anomalies')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel(self.target_column)
        axes[0].set_ylim(global_min, global_max)
        axes[0].legend()

        # Biểu đồ corrected
        axes[1].plot(time_series, corrected, label='Corrected', color='green')
        axes[1].set_title('Corrected Series (Anomalies Reconstructed)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylim(global_min, global_max)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

