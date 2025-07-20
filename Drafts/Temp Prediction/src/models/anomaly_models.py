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
def MyZ_Score(data, data_cols, ax, z_thresh = 3, display = False):
    # Scale dữ liệu
    mean     = data[data_cols].mean()
    std      = data[data_cols].std()
    bound    = (data[data_cols] - mean) / std
    outliers = data[abs(bound) > z_thresh][data_cols]

    if display is True:
        # Vẽ lineplot
        ax[0].plot(data.index, data[data_cols],
                                color     = 'dimgray', 
                                linestyle = '-', 
                                alpha     = 0.7, 
                                label     = f'{data_cols} (Full Series)')
        if not outliers.empty:
            ax[0].scatter(outliers.index, outliers,
                                    color  = 'red', 
                                    label  = 'Outliers', 
                                    marker = 'o')

        ax[0].set_title(f'Z_Score_Modified Outlier Detection - {data_cols}')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Value')
        ax[0].grid(True)
        ax[0].legend()

        # Barplot số outlier theo năm
        if not outliers.empty:
            outlier_counts = outliers.resample('Y').count().astype(int)
            outlier_counts = outlier_counts.reset_index()
            outlier_counts['Year'] = outlier_counts['time'].dt.year

            colors = plt.cm.viridis(np.linspace(0, 1, len(outlier_counts)))
            ax[1].bar(outlier_counts['Year'], outlier_counts[data_cols], color=colors)
            ax[1].set_xticks(outlier_counts['Year'])
            ax[1].set_ylabel('Number of Outliers')
            ax[1].set_title(f'Z_Score_Modified Outlier Count per Year')
            ax[1].set_xlabel('Year')
            ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
            ax[1].text(0.5, 0.5, 'No outliers found.',
                                horizontalalignment = 'center',
                                verticalalignment   = 'center',
                                transform           = ax[1].transAxes,
                                fontsize            = 14)
            ax[1].set_title(f'Z_Score_Modified Outlier Count per Year')
            ax[1].set_xticks([]); ax[1].set_yticks([])

        ax[1].tick_params(axis='x', rotation=45)

    return outliers.index
def MyZ_Score_modified(data, data_cols, ax, modified_z_thresh = 3, display = False):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import median_abs_deviation

    # Scale dữ liệu
    median = data[data_cols].median()
    MAD    = median_abs_deviation(data[data_cols], nan_policy='omit')
    if MAD == 0:
        bound = pd.Series([0] * len(data), index=data.index)
    else:
        bound = 0.6745 * (data[data_cols] - median) / MAD
    outliers = data[abs(bound) > modified_z_thresh][data_cols]

    if display is True:
        # Vẽ lineplot
        ax[0].plot(data.index, data[data_cols],
                                color     = 'dimgray', 
                                linestyle = '-', 
                                alpha     = 0.7, 
                                label     = f'{data_cols} (Full Series)')
        if not outliers.empty:
            ax[0].scatter(outliers.index, outliers,
                                    color  = 'red', 
                                    label  = 'Outliers', 
                                    marker = 'o')
    
        ax[0].set_title(f'Z_Score Outlier Detection - {data_cols}')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Value')
        ax[0].grid(True)
        ax[0].legend()
    
        # Barplot số outlier theo năm
        if not outliers.empty:
            outlier_counts = outliers.resample('Y').count().astype(int)
            outlier_counts = outlier_counts.reset_index()
            outlier_counts['Year'] = outlier_counts['time'].dt.year
    
            colors = plt.cm.viridis(np.linspace(0, 1, len(outlier_counts)))
            ax[1].bar(outlier_counts['Year'], outlier_counts[data_cols], color=colors)
            ax[1].set_xticks(outlier_counts['Year'])
            ax[1].set_ylabel('Number of Outliers')
            ax[1].set_title(f'Z_Score Outlier Count per Year')
            ax[1].set_xlabel('Year')
            ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
            ax[1].text(0.5, 0.5, 'No outliers found.',
                                horizontalalignment = 'center',
                                verticalalignment   = 'center',
                                transform           = ax[1].transAxes,
                                fontsize            = 14)
            ax[1].set_title(f'Z_Score Outlier Count per Year')
            ax[1].set_xticks([]); ax[1].set_yticks([])
    
        ax[1].tick_params(axis='x', rotation=45)

    return outliers.index
def MyIsolationForest(data, data_cols, ax, model, display = False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Scale dữ liệu
    scaler      = StandardScaler()
    np_scaled   = scaler.fit_transform(data[[data_cols]])
    data_scaled = pd.DataFrame(np_scaled, index=data.index, columns=[data_cols])

    # Fit Isolation Forest
    model = model
    model.fit(data_scaled)

    # Predict anomaly
    data_scaled['anomaly'] = model.predict(data_scaled)

    # Lấy index của các outlier
    outlier_indices = np.where(data_scaled['anomaly'] == -1)[0]

    if display is True:
        # Vẽ plot trên dữ liệu gốc
        ax.scatter(data.index, data[data_cols],
                   color  = 'dimgray', 
                   label  = 'Normal', 
                   marker = 'o')

        if not outlier_indices.empty:
            ax.scatter(data.loc[outlier_indices].index, data.loc[outlier_indices, data_cols],
                       color  = 'red', 
                       label  = 'Anomaly', 
                       marker = 'o')
        else:
            print(f"[{data_cols}] No outliers detected.")

        ax.set_title(f'IsolationForest Outlier Detection - {data_cols}')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    # # Trả lại dataframe outliers gốc
    # outliers = data.loc[outlier_indices]

    return outlier_indices

    # Không cần plt.show() ở đây — để người gọi handle show sau khi plot xong
def MyLocalOutlierFactor(data, data_cols, ax, model, display = False):
    from sklearn.neighbors import LocalOutlierFactor
    import numpy as np
    import matplotlib.pyplot as plt

    data_scaled = data[[data_cols]].copy()

    model = model
    y_pred = model.fit_predict(data_scaled)
    outlier_scores = model.negative_outlier_factor_
    is_outlier = y_pred == -1

    if display is True:
        # Vẽ plot
        ax.scatter(data.index, data[data_cols],
                   color  = "dimgray", 
                   marker =  'o', 
                   label  = 'Normal')
        ax.scatter(data.index[is_outlier], data[data_cols][is_outlier],
                   color  = "red", 
                   marker =  'o', 
                   label  = 'Anomaly')
        ax.set_title(f"LOF Outliers Detection - {data_cols}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    # Trả về dict
    outliers_detected = {
        "indices": np.where(is_outlier)[0],
        "scores": outlier_scores[is_outlier]
    }

    # # Trả lại dataframe outliers gốc
    # outliers = data.loc[outliers_detected["indices"]]

    return outliers_detected["indices"]


    # Không cần plt.show() ở đây — để người gọi handle show sau khi plot xong
def MyProphet(data, data_cols, ax, model, display = False, factor = 1.5):
    from prophet import Prophet
    import numpy as np
    import pandas as pd

    # Tách timezone nếu có
    if pd.api.types.is_datetime64tz_dtype(data['time']):
        tz                 = data['time'].dt.tz
        data['time_naive'] = data['time'].dt.tz_convert(None)
    else:
        tz                 = None
        data['time_naive'] = data['time']

    # Chuẩn hóa data cho Prophet
    df_mn = data[['time_naive', data_cols]].rename(columns={'time_naive': 'ds', data_cols: 'y'})

    # Fit model
    # model_mn = model
    model_mn = model
    model_mn.fit(df_mn)

    # Predict
    future_mn   = model_mn.make_future_dataframe(periods=0)
    forecast_mn = model_mn.predict(future_mn)

    # Gán lại timezone nếu có
    if tz is not None:
        forecast_mn['ds'] = forecast_mn['ds'].dt.tz_localize(tz)

    # Merge dự báo với dữ liệu thật
    real_data         = data[['time', data_cols]].rename(columns={'time': 'ds', data_cols: 'y'})
    forecasting_final = pd.merge(forecast_mn[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], real_data, 
                                 how = 'inner', 
                                 on  = 'ds')

    if display is True:
        # Vẽ dữ liệu thực lên ax[0]
        ax[0].scatter(data['time'], data[data_cols],
                   color      = 'dimgray', 
                   linestyle  = '-', 
                   marker     = 'o', 
                   label      = 'Actual Data')

        # Vẽ dự báo yhat lên ax[0]
        ax[0].scatter(forecast_mn['ds'], forecast_mn['yhat'], 
                   color     = 'blue', 
                   linestyle = '--', 
                   linewidth = 1.5, 
                   label     = 'Prophet Forecast')

        # Vẽ vùng confidence interval
        ax[0].fill_between(forecast_mn['ds'],
                           forecast_mn['yhat_lower'],
                           forecast_mn['yhat_upper'],
                           color = 'skyblue', 
                           alpha = 0.3, 
                           label = 'Confidence Interval')

        # Cài đặt biểu đồ
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel(data_cols)
        ax[0].set_title(f'Prophet Forecast - {data_cols}')
        ax[0].grid(True)
        ax[0].legend()


    # Tính error & uncertainty
    forecasting_final['error']       = forecasting_final['y'] - forecasting_final['yhat']
    forecasting_final['uncertainty'] = forecasting_final['yhat_upper'] - forecasting_final['yhat_lower']

    # Phát hiện anomaly
    forecasting_final['anomaly'] = forecasting_final.apply(
        lambda x: 'Anomaly' if (np.abs(x['error']) > factor * x['uncertainty']) else 'Normal', axis=1
    )

    if display is True:
        # Tách anomaly và normal
        colors = {'Anomaly': 'red', 'Normal': 'dimgray'}
        for anomaly_label in ['Normal', 'Anomaly']:
            subset = forecasting_final[forecasting_final['anomaly'] == anomaly_label]
            ax[1].scatter(subset['ds'], subset['y'], 
                       color = colors[anomaly_label],
                       label = anomaly_label)

        # # Optional: plot forecast line và CI
        # ax[1].plot(forecasting_final['ds'], forecasting_final['yhat'],
        #         color='black', linestyle='--', linewidth=1, label='Forecast (yhat)')
        # ax[1].fill_between(forecasting_final['ds'],
        #                 forecasting_final['yhat_lower'],
        #                 forecasting_final['yhat_upper'],
        #                 color='skyblue', alpha=0.3, label='Confidence Interval')

        ax[1].set_title(f'Prophet Outlier Detection - {data_cols}')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel(data_cols)
        ax[1].grid(True)
        ax[1].legend()

    # Trả lại outliers
    outliers = forecasting_final[forecasting_final['anomaly'] == 'Anomaly']
    return outliers.index
def MyAgglomerativeClustering(data, data_cols, ax, model, 
                              display     = False, 
                              window_size = 10, 
                              dendrogram  = False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import kneighbors_graph
    from copy import deepcopy
    from collections import Counter

    # Tách timezone nếu có
    if pd.api.types.is_datetime64tz_dtype(data['time']):
        tz = data['time'].dt.tz
        data['time_naive'] = data['time'].dt.tz_convert(None)
    else:
        tz = None
        data['time_naive'] = data['time']

    # Chọn cột cần clustering
    series = data[data_cols].values.flatten()

    # Chuyển time-series thành sliding windows
    X = np.array([series[i:i + window_size] for i in range(len(series) - window_size)])
    if X.shape[0] == 0:
        print("⚠️ Không đủ dữ liệu để tạo sliding windows.")
        return pd.Series(dtype=int)

    # Standard hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    connectivity = kneighbors_graph(X_scaled, n_neighbors=10, include_self=False)

    # Tạo dendrogram nếu bật
    if dendrogram is True:
        max_dendrogram_samples = 3000
        ax0 = ax[0]

        if X_scaled.shape[0] <= max_dendrogram_samples:
            linked = linkage(X_scaled, method='ward')
        else:
            print(f"📉 Dữ liệu quá lớn ({X_scaled.shape[0]} samples), chỉ vẽ dendrogram mẫu.")
            sample_indices = np.random.choice(X_scaled.shape[0], max_dendrogram_samples, replace=False)
            linked = linkage(X_scaled[sample_indices], method='ward')

        dendrogram(linked, ax=ax0)
        ax0.set_title('Dendrogram (sample)')
        ax0.set_xlabel('Time Series Segments')
        ax0.set_ylabel('Euclidean Distance')

    # Cluster hóa
    agg_clustering = deepcopy(model)
    agg_clustering.set_params(connectivity=connectivity)
    labels = agg_clustering.fit_predict(X_scaled)

    # Tính time_plot và outlier ngay cả khi không display
    time_plot = data['time_naive'].values[:len(labels)]
    series_plot = series[:len(labels)]

    # Phát hiện outlier: cụm nhỏ nhất
    counts          = Counter(labels)
    min_count       = min(counts.values())
    outlier_cluster = [label for label, count in counts.items() if count == min_count][0]
    outlier_indices = np.where(labels == outlier_cluster)[0]
    outliers_time   = time_plot[outlier_indices]
    outliers_value  = series_plot[outlier_indices]

    # Vẽ nếu cần
    if display is True:
        ax1 = ax[1]
        colors = plt.cm.get_cmap('tab10', agg_clustering.n_clusters).colors

        for i in range(agg_clustering.n_clusters):
            ax1.scatter(time_plot[labels == i], series_plot[labels == i],
                        color=colors[i], label=f'Cluster {i}')

        ax1.scatter(outliers_time, outliers_value,
                    color='gold', edgecolor='black', s=100, label='Outliers')

        ax1.set_title(f"Agglomerative Clustering Outlier Detection - {data_cols}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(data_cols)
        ax1.legend()
        ax1.grid(True)

    # Trả về chỉ số outlier
    return outlier_indices
def MyDBSCAN(data, data_cols, ax, model, display = False, window_size=10):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from collections import Counter

    # Tách timezone nếu có
    if pd.api.types.is_datetime64tz_dtype(data['time']):
        tz                 = data['time'].dt.tz
        data['time_naive'] = data['time'].dt.tz_convert(None)
    else:
        tz                 = None
        data['time_naive'] = data['time']

    # Chọn cột cần clustering
    series = data[data_cols].values.flatten()

    # Chuyển time-series thành sliding windows
    X = np.array([series[i:i+window_size] for i in range(len(series) - window_size)])

    # Standard hóa
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit DBSCAN
    labels = model.fit_predict(X_scaled)

    # Detect outlier indices
    outlier_indices = np.where(labels == -1)[0]

    if display is True:
        # Chuyển outlier indices sang time point
        time_plot      = data['time_naive'].values[:len(labels)]
        outliers_time  = time_plot[outlier_indices]
        outliers_value = series[outlier_indices]

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                # Outliers
                ax.scatter(time_plot[labels == label], series[:len(labels)][labels == label],
                           color     = 'red', 
                           edgecolor = 'black', 
                        #    s=100,  
                           label     = 'Anomaly')
            else:
                ax.scatter(time_plot[labels == label], series[:len(labels)][labels == label],
                           color = 'dimgray',
                           label = f'Normal')

        ax.set_title(f"DBSCAN Outlier Detection - {data_cols}")
        ax.set_xlabel("Time")
        ax.set_ylabel(data_cols)
        ax.legend()
        ax.grid(True)

    # Trả outliers về DataFrame
    outlier_indices = np.where(labels == -1)[0]

    return outlier_indices
def MyVanillaAutoencoder(data, data_cols, ax, display = False, window_size=10, epochs=10, batch_size=32):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from sklearn.preprocessing import StandardScaler

    # Tách timezone nếu có
    if pd.api.types.is_datetime64tz_dtype(data['time']):
        tz                 = data['time'].dt.tz
        data['time_naive'] = data['time'].dt.tz_convert(None)
    else:
        tz                 = None
        data['time_naive'] = data['time']

    # Lấy series cần phân tích
    series = data[data_cols].values.flatten()

    # Sliding window
    X = np.array([series[i:i+window_size] for i in range(len(series) - window_size)])

    # Chuẩn hóa
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build autoencoder
    input_dim   = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded     = Dense(32, activation='relu')(input_layer)
    decoded     = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(X_scaled, X_scaled,
                    epochs           = epochs,
                    batch_size       = batch_size,
                    shuffle          = False,
                    validation_split = 0.2,
                    verbose          = 1)  # Tắt log cho gọn

    # Dự đoán và tính lỗi MSE
    reconstructed = autoencoder.predict(X_scaled)
    mse           = np.mean(np.square(X_scaled - reconstructed), axis=1)

    # Ngưỡng phát hiện outlier (VD: top 5%)
    threshold       = np.percentile(mse, 95)
    outlier_indices = np.where(mse > threshold)[0]

    # Lấy timestamp cho từng sample
    time_plot = data['time_naive'].values[:len(mse)]

    if display is True:
        # Vẽ kết quả
        ax.scatter(time_plot, series[:len(mse)], 
                color = 'dimgray', 
                label = 'Normal')
    
        # Outliers
        ax.scatter(time_plot[outlier_indices], series[outlier_indices],
                   color     = 'red', 
                   edgecolor = 'black', 
                #    s=100,  
                   label     = 'Anomaly')
    
        ax.set_title(f'Vanilla Autoencoder Outlier Detection - {data_cols}')
        ax.set_xlabel("Time")
        ax.set_ylabel(data_cols)
        ax.grid(True)
        ax.legend()

    # Trả về vị trí outlier
    return outlier_indices
