import numpy   as np
import pandas  as pd
import seaborn as sns
from matplotlib      import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error

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

def My_R2_SCORE(y_pred, y_true,
                data_cols = "Unknown",
                display   = False, 
                step_size = 24,
                freq      = None,
                output    = "std",
                ax        = None):
    y_train_true = y_true[0]
    y_test_true  = y_true[1]
    y_train_pred = y_pred[0]
    y_test_pred  = y_pred[1]

    if freq:
        y_train_true = y_train_true.resample(freq).mean().interpolate()
        y_train_pred = y_train_pred.resample(freq).mean().interpolate()
        y_test_true  = y_test_true.resample(freq).mean().interpolate()
        y_test_pred  = y_test_pred.resample(freq).mean().interpolate()

    n_train = len(y_train_true)
    n_test  = len(y_test_true)

    if n_train < n_test:
        print("y_train phải lớn hơn y_test")
        return

    step_size = step_size
    if n_test < step_size:
        ws_test   = np.unique([int(i * n_test / 5) for i in range(1, 6) if int(i * n_test / 5) > 0])
        step_size = ws_test[1] - ws_test[0]
        ws_train  = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    else: 
        ws_test  = np.concatenate([np.arange(1, n_test, step_size), [n_test]])
        ws_train = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    window_sizes = np.unique(np.concatenate([ws_test, ws_train]).astype(int))

    train_scores, test_scores, idx = [], [], []
    
    # window_train_true = y_train_true.ewm(span=step_size).mean().dropna()
    # window_train_pred = y_train_pred.ewm(span=step_size).mean().dropna()
    # window_test_true = y_test_true.ewm(span=step_size).mean().dropna()
    # window_test_pred = y_test_pred.ewm(span=step_size).mean().dropna()        
    
    for w in window_sizes:
        # Đảm bảo không vượt giới hạn
        if w > len(y_train_true) or w > len(y_train_pred):
            break

        r2_train = r2_score(y_train_true[:w], y_train_pred[:w])
        train_scores.append(r2_train)

        if w <= len(y_test_true):
            r2_test = r2_score(y_test_true[:w], y_test_pred[:w])
        else:
            r2_test = test_scores[-1] if test_scores else r2_train  # fallback nếu trống

        test_scores.append(r2_test)

        if w - 1 < len(y_train_true.index):
            idx.append(y_train_true.index[w - 1])
  
    if display is True:
        # Vẽ biểu đồ
        ax[0].plot(idx, train_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "blue", 
                   label     = "Training R2")
        ax[0].plot(idx, test_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "orange", 
                   label     = "Test R2")
        ax[0].set_title(f"R2 SCORE - {data_cols}")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("R2 SCORE")
        ax[0].grid(True)
        ax[0].legend()
        # Barplot số outlier theo năm
        bar_width = 0.5
        idx_series = pd.to_datetime(idx)
        years = idx_series.to_series().resample('Y').mean().interpolate().dt.year
        train_scores_yearly = pd.Series(train_scores, index=idx_series).resample('Y').mean().interpolate()
        test_scores_yearly  = pd.Series(test_scores, index=idx_series).resample('Y').mean().interpolate()

        ax[1].bar(years, train_scores_yearly,
                  width = bar_width,
                  label = "Training R2")
        ax[1].bar(years, -test_scores_yearly, 
                  width = bar_width,
                  label = "Test R2")
        ax[1].set_title(f"R2 - {data_cols}")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("R2 SCORE")
        ax[1].set_xticks(years)
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].legend()
    if output == "std":
        return train_scores[-1], test_scores[-1]
    elif output == "list":
        return train_scores[-1], test_scores[-1]

def My_MAE_SCORE(y_pred, y_true,
                 data_cols = "Unknown",
                 display   = False, 
                 step_size = 24,
                 freq      = None,
                 output    = "std",
                 ax        = None):

    y_train_true = y_true[0]
    y_test_true  = y_true[1]
    y_train_pred = y_pred[0]
    y_test_pred  = y_pred[1]

    if freq:
        y_train_true = y_train_true.resample(freq).mean().interpolate()
        y_train_pred = y_train_pred.resample(freq).mean().interpolate()
        y_test_true  = y_test_true.resample(freq).mean().interpolate()
        y_test_pred  = y_test_pred.resample(freq).mean().interpolate()

    n_train = len(y_train_true)
    n_test  = len(y_test_true)

    if n_train < n_test:
        print("y_train phải lớn hơn y_test")
        return

    step_size = step_size
    if n_test < step_size:
        ws_test   = np.unique([int(i * n_test / 5) for i in range(1, 6) if int(i * n_test / 5) > 0])
        step_size = ws_test[1] - ws_test[0]
        ws_train  = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    else: 
        ws_test  = np.concatenate([np.arange(1, n_test, step_size), [n_test]])
        ws_train = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    window_sizes = np.unique(np.concatenate([ws_test, ws_train]).astype(int))
    
    train_scores, test_scores, idx = [], [], []

    # window_train_true = y_train_true.rolling(window=step_size).mean().dropna()
    # window_train_pred = y_train_pred.rolling(window=step_size).mean().dropna()
    # window_test_true = y_test_true.rolling(window=step_size).mean().dropna()
    # window_test_pred = y_test_pred.rolling(window=step_size).mean().dropna()
    
    for w in window_sizes:
        # Đảm bảo không vượt giới hạn
        if w > len(y_train_true) or w > len(y_train_pred):
            break

        mae_train = mean_absolute_error(y_train_true[:w], y_train_pred[:w])
        train_scores.append(mae_train)

        if w <= len(y_test_true):
            mae_test = mean_absolute_error(y_test_true[:w], y_test_pred[:w])
        else:
            mae_test = test_scores[-1] if test_scores else mae_train  # fallback nếu trống

        test_scores.append(mae_test)

        if w - 1 < len(y_train_true.index):
            idx.append(y_train_true.index[w - 1])

    if display:
        # Vẽ lineplot
        ax[0].plot(idx, train_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "blue", 
                   label     = "Training MAE")
        ax[0].plot(idx, test_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "orange", 
                   label     = "Test MAE")
        ax[0].set_title(f"Mean Absolute Error - {data_cols}")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("MAE SCORE")
        ax[0].grid(True)
        ax[0].legend()

        # Barplot số outlier theo năm
        bar_width = 0.5
        idx_series = pd.to_datetime(idx)
        years = idx_series.to_series().resample('Y').mean().interpolate().dt.year
        train_scores_yearly = pd.Series(train_scores, index=idx_series).resample('Y').mean().interpolate()
        test_scores_yearly  = pd.Series(test_scores, index=idx_series).resample('Y').mean().interpolate()

        ax[1].bar(years, train_scores_yearly,
                  width = bar_width,
                  label = "Training MAE")
        ax[1].bar(years, -test_scores_yearly, 
                  width = bar_width,
                  label = "Test MAE")
        ax[1].set_title(f"Mean Absolute Error - {data_cols}")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("MAE SCORE")
        ax[1].set_xticks(years)
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].legend()

    return train_scores[-1], test_scores[-1]
    
def My_MSE_SCORE(y_pred, y_true,
                 data_cols = "Unknown",
                 display   = False, 
                 step_size = 24,
                 freq      = None,
                 output    = "std",
                 ax        = None):

    y_train_true = y_true[0]
    y_test_true  = y_true[1]
    y_train_pred = y_pred[0]
    y_test_pred  = y_pred[1]

    if freq:
        y_train_true = y_train_true.resample(freq).mean().interpolate()
        y_train_pred = y_train_pred.resample(freq).mean().interpolate()
        y_test_true  = y_test_true.resample(freq).mean().interpolate()
        y_test_pred  = y_test_pred.resample(freq).mean().interpolate()

    n_train = len(y_train_true)
    n_test = len(y_test_true)

    if n_train < n_test:
        print("y_train phải lớn hơn y_test")
        return

    step_size = step_size
    if n_test < step_size:
        ws_test   = np.unique([int(i * n_test / 5) for i in range(1, 6) if int(i * n_test / 5) > 0])
        step_size = ws_test[1] - ws_test[0]
        ws_train  = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    else: 
        ws_test  = np.concatenate([np.arange(1, n_test, step_size), [n_test]])
        ws_train = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    window_sizes = np.unique(np.concatenate([ws_test, ws_train]).astype(int))

    train_scores, test_scores, idx = [], [], []

    # window_train_true = y_train_true.rolling(window=step_size).mean().dropna()
    # window_train_pred = y_train_pred.rolling(window=step_size).mean().dropna()
    # window_test_true = y_test_true.rolling(window=step_size).mean().dropna()
    # window_test_pred = y_test_pred.rolling(window=step_size).mean().dropna()

    for w in window_sizes:
        # Đảm bảo không vượt giới hạn
        if w > len(y_train_true) or w > len(y_train_pred):
            break

        mse_train = mean_squared_error(y_train_true[:w], y_train_pred[:w])
        train_scores.append(mse_train)

        if w <= len(y_test_true):
            mse_test = mean_squared_error(y_test_true[:w], y_test_pred[:w])
        else:
            mse_test = test_scores[-1] if test_scores else mse_train  # fallback nếu trống

        test_scores.append(mse_test)

        if w - 1 < len(y_train_true.index):
            idx.append(y_train_true.index[w - 1])

    if display is True:
        # Vẽ biểu đồ
        ax[0].plot(idx, train_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "blue", 
                   label     = "Training MSE")
        ax[0].plot(idx, test_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "orange", 
                   label     = "Test MSE")
        ax[0].set_title(f"Mean Squared Error - {data_cols}")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("MSE SCORE")
        ax[0].grid(True)
        ax[0].legend()
        # Barplot số outlier theo năm
        bar_width = 0.5
        idx_series = pd.to_datetime(idx)
        years = idx_series.to_series().resample('Y').mean().interpolate().dt.year
        train_scores_yearly = pd.Series(train_scores, index=idx_series).resample('Y').mean().interpolate()
        test_scores_yearly  = pd.Series(test_scores, index=idx_series).resample('Y').mean().interpolate()

        ax[1].bar(years, train_scores_yearly,
                  width = bar_width,
                  label = "Training MSE")
        ax[1].bar(years, -test_scores_yearly, 
                  width = bar_width,
                  label = "Test MSE")
        ax[1].set_title(f"Mean Squared Error - {data_cols}")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("MSE SCORE")
        ax[1].set_xticks(years)
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].legend()

    return train_scores[-1], test_scores[-1]

def My_MSLE_SCORE(y_pred, y_true,
                  data_cols = "Unknown",
                  display   = False, 
                  step_size = 24,
                  freq      = None,
                  output    = "std",
                  ax        = None):

    y_train_true = y_true[0]
    y_test_true  = y_true[1]
    y_train_pred = y_pred[0]
    y_test_pred  = y_pred[1]

    if freq:
        y_train_true = y_train_true.resample(freq).mean().interpolate()
        y_train_pred = y_train_pred.resample(freq).mean().interpolate()
        y_test_true  = y_test_true.resample(freq).mean().interpolate()
        y_test_pred  = y_test_pred.resample(freq).mean().interpolate()

    n_train = len(y_train_true)
    n_test  = len(y_test_true)

    if n_train < n_test:
        print("y_train phải lớn hơn y_test")
        return

    step_size = step_size
    if n_test < step_size:
        ws_test   = np.unique([int(i * n_test / 5) for i in range(1, 6) if int(i * n_test / 5) > 0])
        step_size = ws_test[1] - ws_test[0]
        ws_train  = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    else: 
        ws_test  = np.concatenate([np.arange(1, n_test, step_size), [n_test]])
        ws_train = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    window_sizes = np.unique(np.concatenate([ws_test, ws_train]).astype(int))

    train_scores, test_scores, idx = [], [], []

    # window_train_true = y_train_true.rolling(window=step_size).mean().dropna()
    # window_train_pred = y_train_pred.rolling(window=step_size).mean().dropna()
    # window_test_true = y_test_true.rolling(window=step_size).mean().dropna()
    # window_test_pred = y_test_pred.rolling(window=step_size).mean().dropna()

    for w in window_sizes:
        # Đảm bảo không vượt giới hạn
        if w > len(y_train_true) or w > len(y_train_pred):
            break

        msle_train = mean_squared_log_error(y_train_true[:w], y_train_pred[:w])
        train_scores.append(msle_train)

        if w <= len(y_test_true):
            msle_test = mean_squared_log_error(y_test_true[:w], y_test_pred[:w])
        else:
            msle_test = test_scores[-1] if test_scores else msle_train  # fallback nếu trống

        test_scores.append(msle_test)

        if w - 1 < len(y_train_true.index):
            idx.append(y_train_true.index[w - 1])

    if display is True:
        # Vẽ biểu đồ
        ax[0].plot(idx, train_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "blue", 
                   label     = "Training MLSE")
        ax[0].plot(idx, test_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "orange", 
                   label     = "Test MLSE")
        ax[0].set_title(f"Mean Squared Log Error - {data_cols}")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("MLSE SCORE")
        ax[0].grid(True)
        ax[0].legend()
        # Barplot số outlier theo năm
        bar_width = 0.5
        idx_series = pd.to_datetime(idx)
        years = idx_series.to_series().resample('Y').mean().interpolate().dt.year
        train_scores_yearly = pd.Series(train_scores, index=idx_series).resample('Y').mean().interpolate()
        test_scores_yearly  = pd.Series(test_scores, index=idx_series).resample('Y').mean().interpolate()

        ax[1].bar(years, train_scores_yearly,
                  width = bar_width,
                  label = "Training MSLE")
        ax[1].bar(years, -test_scores_yearly, 
                  width = bar_width,
                  label = "Test MSLE")
        ax[1].set_title(f"Mean Squared Log Error - {data_cols}")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("MSLE SCORE")
        ax[1].set_xticks(years)
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].legend()

    return train_scores[-1], test_scores[-1]
    
def My_MAPE_SCORE(y_pred, y_true,
                  data_cols = "Unknown",
                  display   = False, 
                  step_size = 24,
                  freq      = None,
                  output    = "std",
                  ax        = None):
    y_train_true = y_true[0]
    y_test_true  = y_true[1]
    y_train_pred = y_pred[0]
    y_test_pred  = y_pred[1]

    if freq:
        y_train_true = y_train_true.resample(freq).mean().interpolate()
        y_train_pred = y_train_pred.resample(freq).mean().interpolate()
        y_test_true  = y_test_true.resample(freq).mean().interpolate()
        y_test_pred  = y_test_pred.resample(freq).mean().interpolate()

    n_train = len(y_train_true)
    n_test  = len(y_test_true)

    if n_train < n_test:
        print("y_train phải lớn hơn y_test")
        return

    step_size = step_size
    if n_test < step_size:
        ws_test   = np.unique([int(i * n_test / 5) for i in range(1, 6) if int(i * n_test / 5) > 0])
        step_size = ws_test[1] - ws_test[0]
        ws_train  = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    else: 
        ws_test  = np.concatenate([np.arange(1, n_test, step_size), [n_test]])
        ws_train = np.concatenate([np.arange(n_test, n_train, step_size), [n_train]])
    window_sizes = np.unique(np.concatenate([ws_test, ws_train]).astype(int))

    train_scores, test_scores, idx = [], [], []

    # window_train_true = y_train_true.rolling(window=step_size).mean().dropna()
    # window_train_pred = y_train_pred.rolling(window=step_size).mean().dropna()
    # window_test_true = y_test_true.rolling(window=step_size).mean().dropna()
    # window_test_pred = y_test_pred.rolling(window=step_size).mean().dropna()

    for w in window_sizes:
        # Đảm bảo không vượt giới hạn
        if w > len(y_train_true) or w > len(y_train_pred):
            break

        mape_train = mean_absolute_percentage_error(y_train_true[:w], y_train_pred[:w])
        train_scores.append(mape_train)

        if w <= len(y_test_true):
            mape_test = mean_absolute_percentage_error(y_test_true[:w], y_test_pred[:w])
        else:
            mape_test = test_scores[-1] if test_scores else mape_train  # fallback nếu trống

        test_scores.append(mape_test)

        if w - 1 < len(y_train_true.index):
            idx.append(y_train_true.index[w - 1])

    if display is True:
        # Vẽ biểu đồ
        ax[0].plot(idx, train_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "blue", 
                   label     = "Training MAPE")
        ax[0].plot(idx, test_scores, 
                   linestyle = '-', 
                   linewidth = 3, 
                   color     = "orange", 
                   label     = "Test MAPE")
        ax[0].set_title(f"Mean Absolute Percentage Error - {data_cols}")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("MAPE SCORE")
        ax[0].grid(True)
        ax[0].legend()
        # Barplot số outlier theo năm
        bar_width = 0.5
        idx_series = pd.to_datetime(idx)
        years = idx_series.to_series().resample('Y').mean().interpolate().dt.year
        train_scores_yearly = pd.Series(train_scores, index=idx_series).resample('Y').mean().interpolate()
        test_scores_yearly  = pd.Series(test_scores, index=idx_series).resample('Y').mean().interpolate()

        ax[1].bar(years, train_scores_yearly,
                  width = bar_width,
                  label = "Training MAPE")
        ax[1].bar(years, -test_scores_yearly, 
                  width = bar_width,
                  label = "Test MAPE")
        ax[1].set_title(f"Mean Absolute Percentage Error - {data_cols}")
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("MAPE SCORE")
        ax[1].set_xticks(years)
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].legend()

    return train_scores[-1], test_scores[-1]
