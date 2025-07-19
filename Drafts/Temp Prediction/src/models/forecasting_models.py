# # LSTM autoencoder
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
# from tensorflow.keras import Model
# import os
# class LSTMAutoencoder(Model):    
#     def __init__(self):
#         super(LSTMAutoencoder, self).__init__()
#         self.model = None
#         self.history = None
#         self.train_mae_loss = None
#         self.test_mae_loss = None
#         self.anomalies = None
#         self.score_df = None

#     def create_dataset(self, X, y, time_steps):
#         Xs, ys = [], []
#         for i in range(len(X) - time_steps):
#             v = X.iloc[i:(i + time_steps)].values
#             Xs.append(v)        
#             ys.append(y.iloc[i + time_steps])
#         return np.array(Xs), np.array(ys)
    
#     def prepare_data(self, train_data, test_data, target_column, time_steps):
#         """
#         Chuẩn bị dữ liệu cho training và testing
#         """        
#         self.target_column = target_column
#         self.train_data = train_data
#         self.test_data = test_data
#         self.time_steps = time_steps

#         self.scaler = StandardScaler()
#         self.scaler = self.scaler.fit(train_data[[target_column]])

#         # Scale dữ liệu
#         train_data[target_column] = self.scaler.transform(train_data[[target_column]])
#         test_data[target_column] = self.scaler.transform(test_data[[target_column]])
        
#         self.X_train, self.y_train = self.create_dataset(train_data[[target_column]], train_data[target_column], time_steps)
#         self.X_test, self.y_test = self.create_dataset(test_data[[target_column]], test_data[target_column], time_steps)
        
#         return self.X_train, self.y_train, self.X_test, self.y_test
    
#     def build_autoencoder(self):
#         """
#         Xây dựng model model
#         """
#         LSTM_units = 64
#         model = keras.Sequential([
#             layers.LSTM(LSTM_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=False,name='encoder_lstm'),
#             layers.Dropout(0.2, name='encoder_dropout'),
#             layers.RepeatVector(self.X_train.shape[1], name='decoder_repeater'),
#             layers.LSTM(LSTM_units, return_sequences=True, name='decoder_lstm'),
#             layers.Dropout(0.2, name='decoder_dropout'),
#             layers.TimeDistributed(layers.Dense(self.X_train.shape[2]),name='decoder_dense_output')
#         ])

#         model.compile(optimizer='adam', loss='mae')
#         self.model = model
#         return model
    
#     def train(self, epochs=20, batch_size=256, validation_split=0.1, patience=5, model_path=None):
#         """
#         Training model
#         """
#         if self.model is None:
#             self.build_autoencoder()
            
#         cp = tf.keras.callbacks.ModelCheckpoint(
#             f'../models/LSTMAutoencoders/trained_model/{model_path}.keras',
#             monitor='val_loss',
#             save_best_only=True,
#             verbose=1
#         )
#         es = tf.keras.callbacks.EarlyStopping(
#             restore_best_weights=True, 
#             patience=patience
#         )
        
#         self.history = self.model.fit(
#             self.X_train, self.y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             callbacks=[es, cp],
#             shuffle=False
#         )

#     def plot_training_history(self):
#         """
#         Vẽ biểu đồ training history
#         """
#         if self.history is None:
#             print("Model chưa được train!")
#             return
            
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.history.history['loss'], label='Training Loss')
#         plt.plot(self.history.history['val_loss'], label='Validation Loss')
#         plt.legend()

#     def evaluate(self):
#         """
#         Đánh giá model
#         """
#         if self.model is None:
#             print("Model chưa được train!")
#             return
            
#         test_loss = self.model.evaluate(self.X_test, self.y_test)
#         print(f"Test Loss: {test_loss}")
#         return test_loss
    

#     def detect_anomalies(self, full_data, method='std'):
#         """
#         Áp dụng mô hình đã train để detect và reconstruct anomalies trên toàn bộ chuỗi thời gian
#         """
#         if self.model is None:
#             print("Bạn cần train model và tính toán anomaly scores trước!")
#             return       

#         # Scale full data
#         self.scaled_full = full_data.copy()
#         self.scaled_full[self.target_column] = self.scaler.transform(full_data[[self.target_column]])

#         # Tạo tập dữ liệu sequence
#         X_full, y_full = self.create_dataset(self.scaled_full[[self.target_column]], self.scaled_full[self.target_column], self.time_steps)

#         # Dự đoán tái tạo
#         self.X_pred = self.model.predict(X_full)
#         mae_loss = np.mean(np.abs(self.X_pred - X_full), axis=1)
            
#         # Tính threshold từ training
#         if method == 'std':
#             threshold = np.mean(mae_loss) + 3 * np.std(mae_loss)

#         # elif method == 'quantile':
#         #     threshold = np.quantile(mae_loss, 0.97)

#         # elif method == 'gmm':
#         #     from sklearn.mixture import GaussianMixture
#         #     gmm = GaussianMixture(n_components=2, random_state=0)
#         #     gmm.fit(mae_loss)
#         #     # Lấy cluster có mean lớn hơn làm "anomaly"
#         #     means = gmm.means_.flatten()
#         #     anomaly_cluster = np.argmax(means)
#         #     scores = gmm.predict_proba(mae_loss)[:, anomaly_cluster]
#         #     threshold = np.percentile(mae_loss[scores > 0.5], 5)  # rất nhạy cảm
        
#         # Gắn thông tin anomaly vào dataframe
#         self.score_df = full_data[self.time_steps:].copy().reset_index(drop=True)
#         self.score_df['loss'] = mae_loss
#         self.score_df['threshold'] = threshold
#         self.score_df['anomaly'] = self.score_df.loss > self.score_df.threshold
#         anomalies = self.score_df[self.score_df.anomaly == True]
        
#         print(f"Tổng số anomalies phát hiện: {len(anomalies)}")
#         print(f"Tỷ lệ anomalies: {len(anomalies)/len(self.score_df)*100:.2f}%")

#         return anomalies
    
#     def plot_anomaly_scores(self, full_data, time_column='time'):
#         """
#         Vẽ biểu đồ anomaly scores
#         """
#         if self.score_df is None:
#             print("Chưa phát hiện anomalies!")
#             return
        
#         plot_df = self.score_df.copy()
#         plot_df['time'] = full_data[self.time_steps:][time_column].values
#         plt.figure(figsize=(14, 6))
#         sns.lineplot(data=plot_df, x='time', y='loss', label='Test Loss')
#         sns.lineplot(data=plot_df, x='time', y='threshold', label='Threshold')
#         plt.title('Anomaly Scores')
#         plt.xlabel('Time')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     def reconstruct_and_replace_anomalies(self, inverse=True):
#         # Reconstruct anomalies
#         corrected = self.scaled_full[self.time_steps:][[self.target_column]].copy().reset_index(drop=True)
#         anomaly_indices = self.score_df[self.score_df['anomaly']].index

#         for idx in anomaly_indices:
#             corrected_value = self.X_pred[idx, -1, 0]
#             corrected.at[idx, self.target_column] = corrected_value

#         if inverse:
#             corrected[self.target_column] = self.scaler.inverse_transform(corrected[[self.target_column]])

#         # Gắn corrected vào score_df để tiện phân tích
#         self.score_df['corrected'] = corrected[self.target_column]
#         return self.score_df


#     def plot_reconstruct_result(self, time_column='time'):
#         """
#         Vẽ 2 biểu đồ: dữ liệu gốc với anomaly và dữ liệu đã được corrected, dùng cùng scale trục y
#         """
#         if 'corrected' not in self.score_df.columns:
#             print("Thiếu dữ liệu corrected! Hãy chắc chắn đã chạy detect_and_reconstruct_full_series.")
#             return

#         time_series = self.score_df[time_column].values
#         original = self.score_df[self.target_column].values
#         corrected = self.score_df['corrected'].values
#         anomalies = self.score_df[self.score_df['anomaly']]

#         # Xác định min/max để đồng nhất scale
#         global_min = min(original.min(), corrected.min())
#         global_max = max(original.max(), corrected.max())

#         # Vẽ 2 biểu đồ nằm ngang
#         fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

#         # Biểu đồ dữ liệu gốc với anomaly
#         axes[0].plot(time_series, original, label='Original', zorder=1)
#         axes[0].scatter(anomalies[time_column].values,
#                         anomalies[self.target_column].values,
#                         color='red', marker='X', s=60, label='Anomaly', zorder=2)
#         axes[0].set_title('Original Series with Anomalies')
#         axes[0].set_xlabel('Time')
#         axes[0].set_ylabel(self.target_column)
#         axes[0].set_ylim(global_min, global_max)
#         axes[0].legend()

#         # Biểu đồ corrected
#         axes[1].plot(time_series, corrected, label='Corrected', color='green')
#         axes[1].set_title('Corrected Series (Anomalies Reconstructed)')
#         axes[1].set_xlabel('Time')
#         axes[1].set_ylim(global_min, global_max)
#         axes[1].legend()

#         plt.tight_layout()
#         plt.show()
