from sklearn.neighbors import LocalOutlierFactor


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
# def find_outlier(data, method):
#     import numpy as np
#     import pandas as pd

#     # Chỉ lấy các cột số
#     num_cols = data.select_dtypes(include=[np.number]).columns

#     for col in num_cols:
#         if method == "zscore":
#             mean = data[col].mean()
#             std  = data[col].std()
#             lower_bound = mean - 3 * std
#             upper_bound = mean + 3 * std

#         elif method == "iqr":
#             Q1  = data[col].quantile(0.25)
#             Q3  = data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR

#         elif method == "percentile":
#             lower_bound = data[col].quantile(0.01)
#             upper_bound = data[col].quantile(0.99)

#         else:
#             raise ValueError("Method phải là 'zscore', 'iqr' hoặc 'percentile'")

#         # Tìm index outlier
#         outlier_idx = np.where((data[col] < lower_bound) | (data[col] > upper_bound))[0]
#         percent = (len(outlier_idx) / data.shape[0]) * 100
#         print(f"{col:20}: {len(outlier_idx)} outliers ({percent:.2f}%)")
# def remove_outliers(data, method="zscore"):
#     """
#     Remove outliers in numerical columns using Z-score, IQR hoặc Percentile.

#     Parameters:
#     data (pd.DataFrame): The input dataframe.
#     method (str): 'zscore', 'iqr' hoặc 'percentile'

#     Returns:
#     pd.DataFrame: Dataframe after removing outliers.
#     """
#     import numpy as np
#     import pandas as pd

#     original_size = data.shape[0]
#     outlier_idx = set()

#     # Chỉ lấy các cột numeric
#     numeric_cols = data.select_dtypes(include=["number"]).columns

#     for col in numeric_cols:
#         if method == "zscore":
#             mean = data[col].mean()
#             std  = data[col].std()
#             lower_bound = mean - 3 * std
#             upper_bound = mean + 3 * std

#         elif method == "iqr":
#             Q1 = data[col].quantile(0.25)
#             Q3 = data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR

#         elif method == "percentile":
#             lower_bound = data[col].quantile(0.01)
#             upper_bound = data[col].quantile(0.99)

#         else:
#             raise ValueError("Method phải là 'zscore', 'iqr' hoặc 'percentile'")

#         idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
#         outlier_idx.update(idx)

#     data_cleaned = data.drop(index=outlier_idx)

#     after_size = data_cleaned.shape[0]
#     print(f"Original Data Size           : {original_size}")
#     print(f"After Removing Outliers Size : {after_size}")
#     print(f"Data Retained Percentage     : {after_size / original_size:.2%}")

#     return data_cleaned
# def winsorize_outliers(data, 
#                        data_cols  = None, 
#                        method     = "zscore", 
#                        z_thresh   = 3.0, 
#                        iqr_factor = 1.5, 
#                        lower_pct  = 0.01, 
#                        upper_pct  = 0.99, 
#                        verbose    = True):
#     """
#     Capping outliers trong numerical columns.

#     Parameters:
#         data (pd.DataFrame): Dữ liệu gốc.
#         method (str): 'zscore', 'iqr', hoặc 'percentile'
#         data_cols (list): Danh sách cột cần xử lý. Mặc định tất cả numeric columns.
#         z_thresh (float): Ngưỡng z-score nếu method='zscore'
#         iqr_factor (float): Hệ số nhân IQR nếu method='iqr'
#         lower_pct (float): Percentile thấp nếu method='percentile'
#         upper_pct (float): Percentile cao nếu method='percentile'
#         verbose (bool): In log xử lý.

#     Returns:
#         pd.DataFrame: Dữ liệu sau khi capping outlier.
#     """
#     data_capped = data.copy()
#     if data_cols is None:
#         data_cols = data.select_dtypes(include=["number"]).columns

#     for col in data_cols:
#         outlier_count = 0
#         if method == "zscore":
#             mean = data[col].mean()
#             std  = data[col].std()
#             lower, upper = mean - z_thresh * std, mean + z_thresh * std

#         elif method == "iqr":
#             Q1 = data[col].quantile(0.25)
#             Q3 = data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower, upper = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR

#         elif method == "percentile":
#             lower, upper = data[col].quantile(lower_pct), data[col].quantile(upper_pct)

#         else:
#             raise ValueError("Phương pháp phải là 'zscore', 'iqr', hoặc 'percentile'.")

#         mask_lower = data_capped[col] < lower
#         mask_upper = data_capped[col] > upper
#         outlier_count = mask_lower.sum() + mask_upper.sum()

#         data_capped.loc[mask_lower, col] = lower
#         data_capped.loc[mask_upper, col] = upper

#         if verbose:
#             print(f"📊 {col}: {outlier_count} outlier(s) capped bằng {method}")

#     if verbose:
#         print(f"✅ Đã xử lý outliers bằng phương pháp: {method}\n")

#     return data_capped
# def transform_outliers(data,
#                        data_cols = None, 
#                        method    = "log", 
#                        style     = "replace", 
#                        verbose   = True):
#     from scipy import stats
    
#     """
#     Transform numerical feature để giảm ảnh hưởng outlier.

#     Parameters:
#         data (DataFrame): Dữ liệu gốc.
#         method (str): 'log', 'sqrt', 'boxcox', 'yeojohnson'
#         data_cols (list): Danh sách cột cần xử lý. Mặc định tất cả numeric columns.
#         style (str): 'replace' hoặc 'create_new'
#         verbose (bool): In log xử lý.

#     Returns:
#         DataFrame: Dữ liệu sau khi transform
#     """
#     data_transformed = data.copy()
#     if data_cols is None:
#         data_cols = data.select_dtypes(include=["number"]).columns

#     for col in data_cols:
#         try:
#             if method == "log":
#                 if (data[col] < 0).any():
#                     raise ValueError("Log transform yêu cầu giá trị ≥ 0.")
#                 transformed = np.log1p(data[col])

#             elif method == "sqrt":
#                 transformed = np.sqrt(data[col].clip(lower=0))

#             elif method == "boxcox":
#                 transformed, _ = stats.boxcox(data[col].dropna() + 1)
#                 transformed = pd.Series(transformed, index=data[col].dropna().index)

#             elif method == "yeojohnson":
#                 transformed, _ = stats.yeojohnson(data[col].dropna())
#                 transformed = pd.Series(transformed, index=data[col].dropna().index)

#             else:
#                 raise ValueError("Phương pháp phải là 'log', 'sqrt', 'boxcox' hoặc 'yeojohnson'.")

#             if style == "replace":
#                 data_transformed.loc[transformed.index, col] = transformed

#             elif style == "create_new":
#                 new_col = f"{col}_{method}"
#                 data_transformed.loc[transformed.index, new_col] = transformed

#             else:
#                 raise ValueError("Style phải là 'replace' hoặc 'create_new'.")

#             if verbose:
#                 print(f"🔄 {col} transform thành công bằng {method}")

#         except Exception as e:
#             if verbose:
#                 print(f"⚠️ Lỗi xử lý {col}: {e}")

#     return data_transformed
def handle_feature_outliers_over_time(data, data_cols,
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
    import numpy as np
    import sys
    import os
    from copy import deepcopy
    sys.path.append(os.path.abspath("../src"))


    if isinstance(data, pd.DataFrame):
        name = station_name if station_name else "Unknown"
        print(f"🔸 Trạm: {name}")

        start_time_         = pd.to_datetime(start_time) if start_time else None
        end_time_           = pd.to_datetime(end_time)   if end_time else None
        df_filtered         = data.copy()
        df_filtered['time'] = pd.to_datetime(df_filtered['time'], errors='coerce')
        df_filtered         = df_filtered.dropna(subset=['time'])

        if start_time_ is not None:
            df_filtered = df_filtered[df_filtered['time'] >= start_time_]
        if end_time_ is not None:
            df_filtered = df_filtered[df_filtered['time'] <= end_time_]

        if freq:
            df_filtered  = df_filtered.set_index('time')
            numeric_cols = df_filtered.select_dtypes(include='number').columns
            df_filtered  = df_filtered[numeric_cols].resample(freq).mean().interpolate().reset_index()

        df_filtered = df_filtered.set_index('time')
        outlier_idx = list(df_filtered.index) 
        
        df_filtered_temp = df_filtered.copy()
        
        for feature in data_cols:
            if method == "statistic":
                fig, axes = plt.subplots(2, 2, figsize=(20, 10))
                
                from models.anomaly_models import (MyZ_Score,
                                                   MyZ_Score_modified)
                outlier_idx = list(df_filtered.index)                
                for row, sub_method in enumerate(["z_score", "z_score modified"]):
                    if sub_method == "z_score":
                        Z_outlier = MyZ_Score(data      = df_filtered,
                                              data_cols = feature,
                                              display   = False,
                                              z_thresh  = z_thresh,
                                              ax        = list([axes[0,0], axes[0,1]]))
                        print(f"🔹 {feature} (Z_Score, z_thresh={z_thresh}): {len(Z_outlier)} outliers ~ {len(Z_outlier)/len(df_filtered[feature]):.2%}")
                        # if len(Z_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(Z_outlier)))

                    elif sub_method == "z_score modified":
                        ZM_outlier = MyZ_Score_modified(data              = df_filtered,
                                                        data_cols         = feature,
                                                        display           = False,
                                                        modified_z_thresh = modified_z_thresh,
                                                        ax                = list([axes[1,0], axes[1,1]]))
                        print(f"🔹 {feature} (Z_Score_Modified, modified_z_thresh={modified_z_thresh}): {len(ZM_outlier)} outliers ~ {len(ZM_outlier)/len(df_filtered[feature]):.2%}")
                        # if len(ZM_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(ZM_outlier)))
                        
                print(f"~> Outlier detected: {len(outlier_idx)} outliers ~ {len(outlier_idx)/len(df_filtered[feature]):.2%}")
                if display is True:
                    # Vẽ plot trên dữ liệu gốc
                    axes[0,0].plot(df_filtered.index, df_filtered[feature],
                                   color     = 'dimgray', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Normal')                    
                    
                    axes[0,0].scatter(df_filtered.loc[outlier_idx].index, df_filtered.loc[outlier_idx, feature],
                                   color     = 'red', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Anomaly')
                    axes[0,0].set_title(f'Statistic Outlier Handle - Before - {feature}')
                    axes[0,0].set_xlabel('Time')
                    axes[0,0].set_ylabel('Value')
                    axes[0,0].grid(True)
                    axes[0,0].legend()
                    
                    sns.histplot(data = df_filtered, 
                                 x    = feature, 
                                #  y    = feature1,
                                 kde  = True, 
                                 ax   = axes[1,0])
                    axes[1,0].set_title(f"Histplot of {feature} Before - {name}")
                    axes[1,0].set_xlabel(name)
                    axes[1,0].set_ylabel('Count')
                    axes[1,0].grid(True)
                    
                # Handle
                if len(outlier_idx) > 0 and len(outlier_idx)!=len(df_filtered.index):
                    # Gán NaN vào các outlier
                    df_filtered.loc[outlier_idx, feature] = np.nan
                    
                    # Dùng nội suy tuyến tính (linear) để lấp giá trị NaN
                    df_filtered[feature] = df_filtered[feature].interpolate(method='linear')
                else:
                    print("Khong co outlier, skip!!!")
                    continue
                
                if display is True:  
                    # Vẽ plot trên dữ liệu modify
                    axes[0,1].plot(df_filtered.index, df_filtered[feature],
                                   color     = 'dimgray', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Normal')                    
                    
                    axes[0,1].scatter(df_filtered.loc[outlier_idx].index, df_filtered.loc[outlier_idx, feature],
                                   color     = 'red', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Anomaly')
                    axes[0,1].set_title(f'Statistic Outlier Handle - After - {feature}')
                    axes[0,1].set_xlabel('Time')
                    axes[0,1].set_ylabel('Value')
                    axes[0,1].grid(True)
                    axes[0,1].legend()
                    
                    sns.histplot(data = df_filtered, 
                                 x    = feature, 
                                #  y    = feature1,
                                 kde  = True, 
                                 ax   = axes[1,1])
                    axes[1,1].set_title(f"Histplot of {feature} After - {name}")
                    axes[1,1].set_xlabel(name)
                    axes[1,1].set_ylabel('Count')
                    axes[1,1].grid(True)
            elif method == "machine_learning":        
                fig, axes = plt.subplots(2, 2, figsize=(20, 10))
                
                from models.anomaly_models import (MyIsolationForest,
                                                   MyLocalOutlierFactor,
                                                   MyProphet,
                                                   MyAgglomerativeClustering,
                                                   MyDBSCAN,
                                                   MyHDBSCAN,
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
                    if len(MIF_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(MIF_outlier)))
                
                # Option 2
                if models.get("LocalOutlierFactor") is not None:
                    MLOF_model   = models.get("LocalOutlierFactor")
                    MLOF_outlier = MyLocalOutlierFactor(data      = df_filtered,
                                                        data_cols = feature,
                                                        model     = MLOF_model,
                                                        display   = False,
                                                        ax        = None)
                    print(f"🔹 {feature} (LocalOutlierFactor, {MLOF_model}): {len(MLOF_outlier)} outliers ~ {len(MLOF_outlier)/len(df_filtered[feature]):.2%}")
                    if len(MLOF_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(MLOF_outlier)))
                
                # Option 3
                if models.get("Prophet") is not None:
                    MP_model   = deepcopy(models.get("Prophet"))
                    MP_outlier = MyProphet(data      = df_filtered.reset_index(), # Slow!!!
                                           data_cols = feature,
                                           model     = MP_model,
                                           display   = False,
                                           factor    = factor,
                                           ax        = list([None,None]))
                    print(f"🔹 {feature} (Prophet, {MP_model}): {len(MP_outlier)} outliers ~ {len(MP_outlier)/len(df_filtered[feature]):.2%}")
                    if len(MP_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(MP_outlier)))
                
                # Option 4
                if models.get("AgglomerativeClustering") is not None:
                    MAC_model   = models.get("AgglomerativeClustering")
                    MAC_outlier = MyAgglomerativeClustering(data        = df_filtered.reset_index(), # Slow!!!
                                                            data_cols   = feature,
                                                            model       = MAC_model,
                                                            display     = False,
                                                            window_size = window_size,
                                                            dendrogram  = dendrogram,
                                                            ax          = list([None,None]))
                    print(f"🔹 {feature} (AgglomerativeClustering, {MAC_model}): {len(MAC_outlier)} outliers ~ {len(MAC_outlier)/len(df_filtered[feature]):.2%}")
                    if len(MAC_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(MAC_outlier)))

                # Option 5
                if models.get("HDBSCAN") is not None:
                    M_model   = models.get("HDBSCAN")
                    M_outlier = MyHDBSCAN(data        = df_filtered.reset_index(), # Slow!!!
                                          data_cols   = feature,
                                          model       = M_model,
                                          display     = False,
                                          window_size = window_size,
                                          ax          = None)
                    print(f"🔹 {feature} (HDBSCAN, {M_model}): {len(M_outlier)} outliers ~ {len(M_outlier)/len(df_filtered[feature]):.2%}")
                    if len(M_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(M_outlier)))
                
                # Option 6
                if models.get("VanillaAutoencoder") is not None:
                    # MVA_model   = deepcopy(models.get("VanillaAutoencoder"))
                    MVA_outlier = MyVanillaAutoencoder(data        = df_filtered.reset_index(), # Slow!!!
                                                       data_cols   = feature,
                                                       display     = False,
                                                    #    model       = MVA_model,
                                                       ax          = None)
                    print(f"🔹 {feature} (VanillaAutoencoder): {len(MVA_outlier)} outliers ~ {len(MVA_outlier)/len(df_filtered[feature]):.2%}")
                    if len(MVA_outlier) > 0:
                        outlier_idx = list(set(outlier_idx).intersection(set(MVA_outlier)))
                
                print(f"~> Outlier detected: {len(outlier_idx)} outliers ~ {len(outlier_idx)/len(df_filtered[feature]):.2%}")
                if display is True:
                    # Vẽ plot trên dữ liệu gốc
                    axes[0,0].plot(df_filtered.index, df_filtered[feature],
                                   color     = 'dimgray', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Normal')                    
                    
                    axes[0,0].scatter(df_filtered.loc[outlier_idx].index, df_filtered.loc[outlier_idx, feature],
                                   color     = 'red', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Anomaly')
                    axes[0,0].set_title(f'Machine_Learning Outlier Handle - Before - {feature}')
                    axes[0,0].set_xlabel('Time')
                    axes[0,0].set_ylabel('Value')
                    axes[0,0].grid(True)
                    axes[0,0].legend()
                    
                    sns.histplot(data = df_filtered, 
                                 x    = feature, 
                                #  y    = feature1,
                                 kde  = True, 
                                 ax   = axes[1,0])
                    axes[1,0].set_title(f"Histplot of {feature} Before - {name}")
                    axes[1,0].set_xlabel(name)
                    axes[1,0].set_ylabel('Count')
                    axes[1,0].grid(True)
                    
                # Handle
                if len(outlier_idx) > 0 and len(outlier_idx)!=len(df_filtered.index):
                    # Gán NaN vào các outlier
                    df_filtered.loc[outlier_idx, feature] = np.nan

                    # Dùng nội suy tuyến tính (linear) để lấp giá trị NaN
                    df_filtered[feature] = df_filtered[feature].interpolate(method='linear')
                else:
                    print("Khong co outlier, skip!!!")
                    continue
                
                if display is True:  
                    # Vẽ plot trên dữ liệu modify
                    axes[0,1].plot(df_filtered.index, df_filtered[feature],
                                   color     = 'dimgray', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Normal')                    
                    
                    axes[0,1].scatter(df_filtered.loc[outlier_idx].index, df_filtered.loc[outlier_idx, feature],
                                   color     = 'red', 
                                   linestyle = '-', 
                                   alpha     = 0.7, 
                                   label     = 'Anomaly')
                    axes[0,1].set_title(f'Machine_Learning Outlier Handle - After - {feature}')
                    axes[0,1].set_xlabel('Time')
                    axes[0,1].set_ylabel('Value')
                    axes[0,1].grid(True)
                    axes[0,1].legend()
                    
                    sns.histplot(data = df_filtered, 
                                 x    = feature, 
                                #  y    = feature1,
                                 kde  = True, 
                                 ax   = axes[1,1])
                    axes[1,1].set_title(f"Histplot of {feature} After - {name}")
                    axes[1,1].set_xlabel(name)
                    axes[1,1].set_ylabel('Count')
                    axes[1,1].grid(True)
            else:
                raise ValueError(f"Giá trị method không hợp lệ: {method}")

            if display is True:
                plt.suptitle(f'Outlier Handle for {feature} - {name}', fontsize=18)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
            else:
                plt.close(fig)
                
        data_updated         = data.copy()
        data_updated['time'] = pd.to_datetime(data_updated['time'], errors='coerce')
        data_updated         = data_updated.set_index('time')

        # Gán lại các giá trị đã xử lý
        for feature in data_cols:
            if feature in df_filtered.columns:
                data_updated.loc[df_filtered.index, feature] = df_filtered[feature]


        # Trả về DataFrame sau khi cập nhật
        return data_updated.reset_index()
    else:
        raise ValueError("Tham số 'data' hiện tại chỉ hỗ trợ 1 DataFrame.")



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
