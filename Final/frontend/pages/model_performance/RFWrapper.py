from sklearn.ensemble     import RandomForestRegressor
import numpy             as np
import pandas            as pd
from tqdm.auto import tqdm

class RFWrapper:
    """
    Wrapper sklearn-like cho RandomForestRegressor.
    Fix lỗi: Setting an array element with a sequence.
    """

    def __init__(self, model=None, param_dict=None, input_chunk_length=7, output_chunk_length=1):
        if model is not None:
            self.model = model
        elif param_dict is not None:
            self.param_dict = param_dict
            self.model = RandomForestRegressor(**self.param_dict)
        else:
            raise ValueError("Phải truyền model hoặc param_dict")

        self.input_chunk_length  = input_chunk_length
        self.output_chunk_length = output_chunk_length
        
        # Lưu trữ data
        self.last_y_train = None
        self.last_X_train = None
        self.last_y_valid = None
        self.last_X_valid = None 
        self.feature_cols = []

    # ------------------- sklearn API -------------------
    def get_params(self, deep=True):
        return {"input_chunk_length"  : self.input_chunk_length,
                "output_chunk_length" : self.output_chunk_length,
                **self.model.get_params()}

    def set_params(self, **params):
        if "input_chunk_length" in params:
            self.input_chunk_length = params.pop("input_chunk_length")
        if "output_chunk_length" in params:
            self.output_chunk_length = params.pop("output_chunk_length")
        
        self.param_dict.update(**params)
        self.model = RandomForestRegressor(**self.param_dict)
        
        # Lưu trữ data
        self.last_y_train = None
        self.last_X_train = None
        self.last_y_valid = None
        self.last_X_valid = None 
        self.feature_cols = []
        
        return self

    # ------------------- FIT -------------------
    def fit(self, X, y, X_valid=None, y_valid=None, verbose=True, fi=False):
        # 1. Chuẩn hóa về Pandas để dễ xử lý
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        self.feature_cols = X.columns.tolist()

        # 2. Lưu trữ dữ liệu gốc
        self.last_y_train = y
        self.last_X_train = X
        self.last_y_valid = y_valid
        self.last_X_valid = X_valid

        # 3. Chuẩn bị dữ liệu Numpy
        # Ép kiểu float để tránh lỗi object (như bài trước)
        X_vals = X.values.astype(np.float64)
        y_vals = y.values.astype(np.float64).flatten()
        
        # --- FIX LỖI INDEX ERROR TẠI ĐÂY ---
        # Kiểm tra độ lệch độ dài
        len_x = len(X_vals)
        len_y = len(y_vals)
        
        if len_x != len_y:
            print(f"⚠️ Cảnh báo: Kích thước X ({len_x}) và y ({len_y}) không khớp!")
            # Lấy độ dài chung nhỏ nhất để an toàn
            min_len = min(len_x, len_y)
            X_vals = X_vals[:min_len]
            y_vals = y_vals[:min_len]
            n_samples = min_len
        else:
            n_samples = len_x
        # -----------------------------------

        n_features = X_vals.shape[1]
        
        # Tính toán index loop
        start_idx = self.input_chunk_length
        end_idx = n_samples - self.output_chunk_length + 1
        num_train_samples = end_idx - start_idx
        
        if num_train_samples <= 0:
            raise ValueError(f"Dữ liệu quá ngắn ({n_samples}) so với input ({self.input_chunk_length}) + output ({self.output_chunk_length}).")

        # Pre-allocate arrays
        input_dim = (self.input_chunk_length * n_features) + self.input_chunk_length
        X_train_rf = np.zeros((num_train_samples, input_dim), dtype=np.float64)
        
        if self.output_chunk_length == 1:
            y_train_rf = np.zeros((num_train_samples,), dtype=np.float64)
        else:
            y_train_rf = np.zeros((num_train_samples, self.output_chunk_length), dtype=np.float64)

        if verbose: 
            print(f">> Tạo Rolling Data: Samples={num_train_samples} | Input={self.input_chunk_length} | Output={self.output_chunk_length}")

        # 4. Tạo dữ liệu Rolling
        idx = 0
        for i in range(start_idx, end_idx):
            # --- Input ---
            # X lags
            x_window = X_vals[i - self.input_chunk_length : i].flatten()
            # y lags
            y_window = y_vals[i - self.input_chunk_length : i]
            
            X_train_rf[idx] = np.concatenate([x_window, y_window])
            
            # --- Target ---
            target = y_vals[i : i + self.output_chunk_length]
            
            # (Safety check thừa nhưng an toàn)
            if len(target) == 0:
                raise IndexError(f"Lỗi logic: Target rỗng tại index {i}. Check len(y)={len(y_vals)}")

            if self.output_chunk_length == 1:
                y_train_rf[idx] = target[0] 
            else:
                y_train_rf[idx] = target
                
            idx += 1

        # 5. Fit
        if verbose: print(f">> Fitting RandomForest...")
        self.model.fit(X_train_rf, y_train_rf)
        
        if fi:
            self._calculate_feature_importance(n_features)
                
        return self

    def _calculate_feature_importance(self, n_features_X):
        """Tách logic FI ra hàm riêng cho gọn"""
        try:
            all_lags_importance = self.model.feature_importances_
            input_len = self.input_chunk_length
            
            # Chỉ tính phần X lags (bỏ qua phần y lags ở đuôi vector input)
            # Vector input cấu trúc: [X_lag_1...X_lag_N, y_lag_1...y_lag_N]
            # Độ dài phần X là: n_features_X * input_len
            n_lags_X = n_features_X * input_len
            
            feature_importance_values = np.zeros(n_features_X, dtype=float)
            
            for i in range(n_lags_X):
                col_index = i % n_features_X
                feature_importance_values[col_index] += all_lags_importance[i]
            
            self.feature_importances_ = feature_importance_values
        except Exception as e:
            print(f"Warning calculating FI: {e}")

    
    def predict(self, X, use_valid_history=True, verbose=True):
        """
        Dự báo tương lai (Rolling/Recursive Forecast).
        Logic: Loop là bắt buộc vì y_t phụ thuộc vào dự báo y_{t-1}.
        """
        icl = self.input_chunk_length
        
        # 1. Build History
        y_hist = self.last_y_train.values.astype(float).flatten()
        X_hist = self.last_X_train.values.astype(float)
        
        if use_valid_history and self.last_y_valid is not None:
            y_hist = np.concatenate([y_hist, self.last_y_valid.values.astype(float).flatten()])
            X_hist = np.vstack([X_hist, self.last_X_valid.values.astype(float)])
        
        # Lấy icl điểm cuối cùng làm điểm tựa khởi đầu
        curr_y_buffer = y_hist[-icl:].tolist()
        
        # X_future: Dữ liệu đặc trưng tương lai người dùng truyền vào
        X_future = X.values.astype(float) if hasattr(X, "values") else np.array(X, dtype=float)
        # Nối X lịch sử và X tương lai để dễ lấy cửa sổ sliding
        X_full = np.vstack([X_hist, X_future])
        
        n_steps = len(X_future)
        preds = []
        n_features = X_full.shape[1]

        # 2. Vòng lặp dự báo (Recursive)
        for i in tqdm(range(n_steps)):
            # Lấy cửa sổ X từ X_full (tính toán dựa trên index của X_future trong X_full)
            start_x_idx = len(X_hist) + i - icl
            end_x_idx = len(X_hist) + i
            x_window = X_full[start_x_idx:end_x_idx].flatten()
            
            # Lấy cửa sổ y từ buffer (luôn chứa icl điểm gần nhất)
            y_window = np.array(curr_y_buffer[-icl:])
            
            # Kết hợp features
            feat = np.concatenate([x_window, y_window]).reshape(1, -1)
            
            # Predict
            y_hat = self.model.predict(feat)[0]
            
            # Nếu model trả về mảng (multi-output), lấy phần tử đầu tiên 
            # (hoặc tùy chỉnh nếu bạn muốn dự báo đa bước thực sự)
            if isinstance(y_hat, np.ndarray): y_hat = y_hat[0]
            
            preds.append(y_hat)
            curr_y_buffer.append(y_hat) # Đưa kết quả dự báo vào lại buffer để làm lag cho bước sau
            
        return np.array(preds)

    def predict_history(self, type="valid", verbose=True):
        """
        Dự báo song song trên toàn bộ tập dữ liệu (Batch Prediction).
        Logic: Gom tất cả lags thành 1 ma trận duy nhất rồi predict 1 lần.
        """
        # 1. Chuẩn bị dữ liệu đầu vào
        if type == "train":
            X_vals = self.last_X_train.values.astype(np.float64)
            y_vals = self.last_y_train.values.astype(np.float64).flatten()
            start_idx = self.input_chunk_length
        else:
            # Đối với Valid, cần nối thêm ICL điểm cuối của Train để dự báo được điểm đầu tiên
            icl = self.input_chunk_length
            X_vals = np.vstack([self.last_X_train.values[-icl:], self.last_X_valid.values]).astype(np.float64)
            y_vals = np.concatenate([self.last_y_train.values[-icl:], self.last_y_valid.values]).astype(np.float64).flatten()
            start_idx = icl

        n_samples = len(X_vals)
        n_features = X_vals.shape[1]
        icl = self.input_chunk_length
        
        # 2. Tính toán số lượng mẫu dự báo được
        # start_idx đến cuối chuỗi
        predict_indices = range(start_idx, n_samples - self.output_chunk_length + 1)
        num_forecasts = len(predict_indices)
        
        if num_forecasts <= 0: return np.array([])

        # 3. TẠO MA TRẬN LAGGING (GIỐNG HÀM FIT)
        input_dim = (icl * n_features) + icl
        X_batch = np.zeros((num_forecasts, input_dim), dtype=np.float64)
        
        for i, curr_idx in enumerate(predict_indices):
            x_window = X_vals[curr_idx - icl : curr_idx].flatten()
            y_window = y_vals[curr_idx - icl : curr_idx]
            X_batch[i] = np.concatenate([x_window, y_window])

        # 4. DỰ BÁO 1 LẦN DUY NHẤT
        if verbose: print(f">> Batch Predict {type}: {num_forecasts} samples")
        preds = self.model.predict(X_batch)
        
        # Flatten kết quả nếu là single-output
        return preds.flatten() if self.output_chunk_length == 1 else preds