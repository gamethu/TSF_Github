import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from __future__ import print_function

# np.random.seed(1337)  # for reproducibility

from keras.preprocessing     import sequence
from keras.models            import Sequential
from keras.layers            import Dense, Dropout, Activation
from keras.layers            import Convolution1D, MaxPooling1D, Flatten
from keras                   import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from keras.layers            import LSTM
from keras.callbacks         import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics         import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Đọc dữ liệu từ AMZN.csv
df          = pd.read_csv('stock/AMZN.csv')
df['Date']  = pd.to_datetime(df['Date'], format="%m/%d/%Y")
df          = df.sort_values(by='Date')
df['Price'] = df['Price'].astype(float)

# Chuẩn bị dữ liệu
df1       = pd.DataFrame(df, columns=['Date', 'Price'])
df1.index = df1['Date']
df1.drop('Date', axis=1, inplace=True)

data       = df1.values
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data  = data[train_size:]

# Chuẩn hóa dữ liệu với MinMaxScaler
scaler   = MinMaxScaler(feature_range=(0, 1))
sc_train = scaler.fit_transform(train_data)
sc_test  = scaler.transform(test_data)

# Tạo dữ liệu huấn luyện với window size 60
window_size = 10
X_train, y_train = [], []
for i in range(window_size, len(train_data)):
    X_train.append(sc_train[i - window_size:i, 0])
    y_train.append(sc_train[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape dữ liệu cho CNN [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

# Tạo dữ liệu kiểm tra
X_test, y_test = [], []
for i in range(window_size, len(test_data)):
    X_test.append(sc_test[i - window_size:i, 0])
    y_test.append(sc_test[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

# Định nghĩa kích thước đầu ra của LSTM
lstm_output_size = 70

# Xây dựng mô hình CNN+LSTM
cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=(window_size, 1)))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="linear"))

# Biên dịch mô hình
cnn.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mse'])

# Định nghĩa callbacks
checkpointer   = ModelCheckpoint(filepath       = "logs/cnn-lstm/checkpoint-{epoch:02d}.keras", 
                                 verbose        = 1, 
                                 save_best_only = True,
                                 monitor        = 'val_loss',
                                 mode           = 'min')
csv_logger     = CSVLogger('logs/cnn-lstm/cnntrainanalysis1.csv', 
                           separator = ',', 
                           append    = False)
early_stopping = EarlyStopping(monitor              = 'val_loss', 
                               patience             = 20, 
                               restore_best_weights = True)

# Huấn luyện mô hình
cnn.fit(X_train, y_train, 
        epochs          = 200, 
        validation_data = (X_test, y_test),
        callbacks       = [checkpointer, csv_logger, early_stopping])

# Lưu mô hình
cnn.save("logs/cnn-lstm/cnn_model.keras")

# Dự đoán
y_train_predict = cnn.predict(X_train)
y_test_predict  = cnn.predict(X_test)


# Tính chỉ số trên dữ liệu chuẩn hóa (trước inverse transform)
def print_metrics(y_true, y_pred, dataset_name):
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f'\n---📘 {dataset_name} ---')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'MAPE: {mape:.6f}%')
    print(f'R2 Score: {r2:.6f}')


# In kết quả trên dữ liệu chuẩn hóa
print_metrics(y_train, y_train_predict, 'Tập huấn luyện (Train)')
print_metrics(y_test, y_test_predict, 'Tập kiểm tra (Test)')

# Vẽ biểu đồ (giữ nguyên như trước)
train_data1 = df1[window_size:train_size].copy()
test_data1  = df1[train_size + window_size:].copy()

y_train_predict_inverse = scaler.inverse_transform(y_train_predict)
y_test_predict_inverse  = scaler.inverse_transform(y_test_predict)
y_train_inverse         = scaler.inverse_transform(y_train)
y_test_inverse          = scaler.inverse_transform(y_test)

plt.figure(figsize = (24, 8))
plt.plot(df1.index, df1['Price'], label='Giá thực tế', color='red')
train_data1['Dự đoán'] = y_train_predict_inverse.flatten()
plt.plot(train_data1.index, train_data1['Dự đoán'], label='Giá dự đoán train', color='green')
test_data1['Dự đoán'] = y_test_predict_inverse.flatten()
plt.plot(test_data1.index, test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')
plt.title('CNN+LSTM - So sánh giá dự báo và giá thực tế')
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa (USD)')
plt.legend()
plt.savefig('price_comparison_cnn_lstm.png')
plt.close()