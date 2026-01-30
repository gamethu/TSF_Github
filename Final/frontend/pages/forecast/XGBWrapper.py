from xgboost              import XGBRegressor
import numpy             as np
import pandas            as pd
from tqdm.auto import tqdm

class XGBWrapper:
    """
    Wrapper sklearn-like cho XGBRegressor.
    Fix lỗi: Setting an array element with a sequence.
    """

    def __init__(self, model=None, param_dict=None, input_chunk_length=7, output_chunk_length=1):
        if model is not None:
            self.model = model
        elif param_dict is not None:
            self.param_dict = param_dict
            self.model = XGBRegressor(**self.param_dict)
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
        self.model = XGBRegressor(**self.param_dict)
        
        # Lưu trữ data
        self.last_y_train = None
        self.last_X_train = None
        self.last_y_valid = None
        self.last_X_valid = None 
        self.feature_cols = []
        
        return self

    # ------------------- HELPER -------------------
    def _create_lagged_data(self, X_vals, y_vals):
        """Chuyển đổi dữ liệu thô sang dạng ma trận Lagging (Rolling Window)"""
        icl = self.input_chunk_length
        ocl = self.output_chunk_length
        n_samples = len(X_vals)
        n_features = X_vals.shape[1]
        
        start_idx = icl
        end_idx = n_samples - ocl + 1
        num_samples_rf = end_idx - start_idx
        
        if num_samples_rf <= 0:
            return None, None

        # Pre-allocate
        input_dim = (icl * n_features) + icl
        X_rf = np.zeros((num_samples_rf, input_dim), dtype=np.float64)
        y_rf = np.zeros((num_samples_rf,) if ocl == 1 else (num_samples_rf, ocl), dtype=np.float64)

        for i, curr_idx in enumerate(range(start_idx, end_idx)):
            # Input: [X_lags, y_lags]
            x_window = X_vals[curr_idx - icl : curr_idx].flatten()
            y_window = y_vals[curr_idx - icl : curr_idx]
            X_rf[i] = np.concatenate([x_window, y_window])
            
            # Target
            target = y_vals[curr_idx : curr_idx + ocl]
            X_rf[i] = np.concatenate([x_window, y_window])
            y_rf[i] = target[0] if ocl == 1 else target
            
        return X_rf, y_rf

    # ------------------- FIT -------------------
    def fit(self, X, y, X_valid=None, y_valid=None, verbose=True, fi=False):
        # 1. Chuẩn hóa Pandas -> Numpy
        X_vals = X.values.astype(np.float64) if hasattr(X, "columns") else X.astype(np.float64)
        y_vals = y.values.astype(np.float64).flatten() if hasattr(y, "name") else y.astype(np.float64).flatten()
        self.feature_cols = X.columns.tolist() if hasattr(X, "columns") else [f'feat_{i}' for i in range(X.shape[1])]

        # Lưu trữ để predict sau này
        self.last_y_train, self.last_X_train = y, X
        self.last_y_valid, self.last_X_valid = y_valid, X_valid

        # 2. Tạo dữ liệu Rolling cho tập Train
        if verbose: print(f">> Tạo Rolling Data Train...")
        X_train_rf, y_train_rf = self._create_lagged_data(X_vals, y_vals)

        # 3. Tạo dữ liệu Rolling cho tập Valid (nếu có)
        eval_set = []
        if X_valid is not None and y_valid is not None:
            if verbose: print(f">> Tạo Rolling Data Valid...")
            X_v_vals = X_valid.values.astype(np.float64) if hasattr(X_valid, "columns") else X_valid.astype(np.float64)
            y_v_vals = y_valid.values.astype(np.float64).flatten() if hasattr(y_valid, "name") else y_valid.astype(np.float64).flatten()
            
            # Để valid điểm đầu tiên, cần nối thêm ICL điểm cuối của train
            icl = self.input_chunk_length
            X_v_combined = np.vstack([X_vals[-icl:], X_v_vals])
            y_v_combined = np.concatenate([y_vals[-icl:], y_v_vals])
            
            X_val_rf, y_val_rf = self._create_lagged_data(X_v_combined, y_v_combined)
            if X_val_rf is not None:
                eval_set = [(X_val_rf, y_val_rf)]

        # 4. Fit
        if verbose: print(f">> Fitting XGBoost (Samples={len(X_train_rf)})...")
        self.model.fit(X=X_train_rf, y=y_train_rf, eval_set=eval_set, verbose=0)
        
        if fi:
            self._calculate_feature_importance(X_vals.shape[1])
                
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

    # ... (Giữ nguyên phần predict và predict_history cũ của bạn) ...
    # Bạn chỉ cần copy lại phần predict/predict_history vào đây
    
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

    # ------------------- PREDICT HISTORY -------------------
    def predict_history(self, type="valid", verbose=True):
        """Dự báo Batch trên lịch sử (Train hoặc Valid)"""
        icl = self.input_chunk_length
        
        # 1. Chuẩn bị Data thô dựa theo type
        if type == "train":
            X_vals = self.last_X_train.values.astype(np.float64)
            y_vals = self.last_y_train.values.astype(np.float64).flatten()
        else:
            # Valid: Nối ICL điểm cuối Train để bắt được điểm đầu Valid
            X_vals = np.vstack([self.last_X_train.values[-icl:], self.last_X_valid.values]).astype(np.float64)
            y_vals = np.concatenate([self.last_y_train.values[-icl:], self.last_y_valid.values]).astype(np.float64).flatten()

        # 2. Tạo ma trận Lagging
        X_batch, _ = self._create_lagged_data(X_vals, y_vals)
        
        if X_batch is None: return np.array([])

        # 3. Predict 1 lần duy nhất
        if verbose: print(f">> Batch Predict {type}: {len(X_batch)} samples")
        preds = self.model.predict(X_batch)
        
        return preds.flatten() if self.output_chunk_length == 1 else preds