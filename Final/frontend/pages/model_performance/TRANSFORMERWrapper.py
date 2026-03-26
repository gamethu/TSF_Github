from xgboost              import XGBRegressor
import numpy             as np
import pandas            as pd
from tqdm.auto import tqdm
from darts.models.forecasting.transformer_model import TransformerModel
from pytorch_lightning.callbacks import Callback
from darts import TimeSeries
import torch
from pytorch_lightning import Trainer
from darts.metrics import rmse

class TRANSFORMERWrapper:
    """
    Wrapper sklearn-like cho TransformerModel (Darts 0.35.0)
    Tách biệt lưu trữ Train và Valid set.
    """

    def __init__(self, model=None, param_dict=None):
        if model is not None:
            self.model = model
            self.param_dict = getattr(model, '_model_params', {}).copy()
        elif param_dict is not None:
            self.model = TransformerModel(**param_dict)
            self.param_dict = param_dict
        else:
            raise ValueError("Phải truyền model hoặc param_dict")

        self.feature_cols = None
        self.freq = None
        
        # 🚨 LƯU TRỮ TÁCH BIỆT TRAIN VÀ VALID
        self.last_y_train_ts = None
        self.last_X_train_ts = None
        self.last_y_valid_ts = None
        self.last_X_valid_ts = None 
        
        self.feature_importances_ = None
        self.history = {}
   
    # ------------------- sklearn API -------------------
    def get_params(self, deep=True):
        return self.param_dict

    def set_params(self, **params):
        self.param_dict.update(params)
        self.model = TransformerModel(**self.param_dict)
        self.feature_importances_ = None
        self.last_y_train_ts      = None
        self.last_X_train_ts      = None
        self.last_y_valid_ts      = None
        self.last_X_valid_ts      = None
        self.history              = {}
        return self
    
    
    class LossLogger(Callback):
        def __init__(self):
            self.train_loss = []
            self.val_loss   = []

        # Bắt sự kiện cuối mỗi epoch huấn luyện
        def on_train_epoch_end(self, trainer, pl_module):
            # Lấy giá trị loss từ metrics (Darts thường log key là "train_loss")
            loss = trainer.callback_metrics.get("train_loss")
            if loss is not None:
                self.train_loss.append(loss.item())

        # Bắt sự kiện cuối mỗi epoch validation
        def on_validation_epoch_end(self, trainer, pl_module):
            loss = trainer.callback_metrics.get("val_loss")
            if loss is not None:
                self.val_loss.append(loss.item())
                
    # ------------------- FIT -------------------
    def fit(self, X, y, X_valid=None, y_valid=None, verbose=True, fi=False, loss=False):
    
        # 1. Store feature columns và Chuẩn hóa X/y về DataFrame/Series
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
        self.feature_cols = X.columns.tolist()
        
        # 2. Convert to TimeSeries (Bắt buộc để lấy freq)
        y_ts = TimeSeries.from_series(y)
        self.freq = y_ts.freq_str 
        
        if self.freq is None:
            raise ValueError("Lỗi: Không thể xác định tần suất (frequency) của TimeSeries. Đảm bảo Index là DatetimeIndex liên tục.")
    
        # Đảm bảo X có DatetimeIndex
        if not isinstance(X.index, pd.DatetimeIndex):
             X.index = pd.date_range(start=y.index[0], periods=len(X), freq=self.freq)
        
        X_ts = TimeSeries.from_dataframe(X, freq=self.freq)
        
        # Xử lý Validation Set
        y_valid_ts, X_valid_ts = None, None
        if X_valid is not None and y_valid is not None:
            if isinstance(X_valid, np.ndarray):
                 X_valid = pd.DataFrame(X_valid, columns=self.feature_cols)
    
            y_valid_ts = TimeSeries.from_series(y_valid, freq=self.freq)
            
            if not isinstance(X_valid.index, pd.DatetimeIndex):
                 X_valid.index = pd.date_range(start=y_valid.index[0], periods=len(X_valid), freq=self.freq)
                 
            X_valid_ts = TimeSeries.from_dataframe(X_valid, freq=self.freq)
        
        # =========================================================
        # 🚨 CẬP NHẬT: GỘP CALLBACKS TỪ PL_TRAINER_KWARGS
        # =========================================================
        loss_logger = self.LossLogger()
        
        # Lấy pl_trainer_kwargs và trích xuất callbacks người dùng định nghĩa
        pl_kwargs = self.param_dict.get("pl_trainer_kwargs", {}).copy()
        user_callbacks = pl_kwargs.pop("callbacks", [])
        if not isinstance(user_callbacks, list):
            user_callbacks = [user_callbacks]
            
        # Gộp EarlyStopping (user_callbacks) với loss_logger
        current_callbacks = user_callbacks + ([loss_logger] if loss==True else [])

        device = torch.cuda.is_available()
        trainer = None
        if device:
            trainer = Trainer(accelerator         = 'gpu',
                              devices             = torch.cuda.device_count(),
                              max_epochs          = self.param_dict.get("n_epochs"),
                              precision           = "64-true",
                              enable_progress_bar = True,
                              logger              = True,
                              callbacks           = current_callbacks, # Thêm callbacks ở đây
                              **pl_kwargs)         # Gộp các tham số còn lại
            print(f"Sử dụng chế độ {torch.cuda.device_count()} GPU cho dự báo.")
        else:
            trainer = Trainer(accelerator         = 'cpu',
                              devices             = 1,
                              max_epochs          = self.param_dict.get("n_epochs"),
                              precision           = "64-true",
                              enable_progress_bar = True,
                              logger              = True,
                              callbacks           = current_callbacks, # Thêm callbacks ở đây
                              **pl_kwargs)         # Gộp các tham số còn lại
            print(f"Sử dụng chế độ CPU cho dự báo.")
        
        
        # 3. Fit Model
        self.model.fit(
            series              = y_ts,
            past_covariates     = X_ts,
            val_series          = y_valid_ts,
            val_past_covariates = X_valid_ts,
            trainer             = trainer,
            verbose             = verbose
        )
    
        self.history["train_loss"] = loss_logger.train_loss
        self.history["val_loss"]   = loss_logger.val_loss
        
        if verbose:
            print(f"Train Loss (Final): {loss_logger.train_loss[-1] if loss_logger.train_loss else 'N/A'}")
            print(f"Val Loss (Final)  : {loss_logger.val_loss[-1] if loss_logger.val_loss else 'N/A'}")
    
        # 4. 🚨 Lưu lịch sử TÁCH BIỆT Train và Valid
        self.last_y_train_ts = y_ts
        self.last_X_train_ts = X_ts
        self.last_y_valid_ts = y_valid_ts
        self.last_X_valid_ts = X_valid_ts
    
        if fi:
            try:
                if verbose:
                    print(">> Đang tính Feature Importance (Permutation Method)...")

                # A. Xác định tập dữ liệu để đánh giá (ưu tiên Valid Set)
                if X_valid_ts is not None and y_valid_ts is not None:
                    # 🚨 Logic nối chuỗi: Lấy đoạn cuối Train + Valid để model có đủ context
                    start_lookback = y_ts.end_time() - pd.Timedelta(self.model.input_chunk_length - 1, unit=self.freq)

                    eval_series = y_ts.slice(start_lookback, y_ts.end_time()).concatenate(y_valid_ts, axis=0)
                    eval_covariates = X_ts.slice(start_lookback, X_ts.end_time()).concatenate(X_valid_ts, axis=0)
                else:
                    # Nếu không có Valid, dùng Train
                    eval_series = y_ts
                    eval_covariates = X_ts

                # B. Tính Baseline RMSE (Dự báo chuẩn không xáo trộn)
                baseline_preds = self.model.historical_forecasts(
                    series           = eval_series,
                    past_covariates  = eval_covariates,
                    start            = self.model.input_chunk_length,
                    forecast_horizon = 1,
                    retrain          = False,
                    verbose          = False,
                    # trainer         = trainer
                )

                # Cắt target thực tế khớp với đoạn dự báo
                intersect_target = eval_series.slice(baseline_preds.start_time(), baseline_preds.end_time())
                baseline_score = rmse(intersect_target, baseline_preds)

                # C. Loop Permutation (Xáo trộn từng feature)
                importances = []
                df_cov_orig = pd.DataFrame(eval_covariates.values(),
                                       index=eval_covariates.time_index,
                                       columns=self.feature_cols)

                for col in self.feature_cols:
                    # Copy và shuffle cột col
                    df_shuffled = df_cov_orig.copy()
                    df_shuffled[col] = np.random.permutation(df_shuffled[col].values)
                    ts_shuffled = TimeSeries.from_dataframe(df_shuffled, freq=self.freq)

                    # Dự báo lại
                    shuffled_preds = self.model.historical_forecasts(
                        series           = eval_series,
                        past_covariates  = ts_shuffled,
                        start            = self.model.input_chunk_length,
                        forecast_horizon = 1,
                        retrain          = False,
                        verbose          = False,
                        # trainer        = trainer
                    )

                    score = rmse(intersect_target, shuffled_preds)
                    # Importance = Lỗi tăng thêm
                    importances.append(max(0, score - baseline_score))

                # D. Normalize về 1
                imp_arr = np.array(importances)
                if imp_arr.sum() > 0:
                    self.feature_importances_ = imp_arr / imp_arr.sum()
                else:
                    self.feature_importances_ = np.zeros(len(self.feature_cols))

                if verbose:
                    print(">> Tính Feature Importance hoàn tất.")

            except Exception as e:
                if verbose:
                    print(f"Cảnh báo: Lỗi khi tính Feature Importance. Đảm bảo NBEATSTExplainer đã được cài đặt đúng cách và dữ liệu đủ dài. Chi tiết: {e}")
                self.feature_importances_ = None
    
        return self

    # ------------------- PREDICT -------------------
    def predict(self, 
                X, 
                use_valid_history = True, 
                verbose           = True, 
                n_jobs            = -1, 
                rolling: bool     =  True):
        
        import warnings, sys, copy, torch, logging
        warnings.filterwarnings("ignore")
    
        # ===== 0. TẮT TOÀN BỘ LOG LIGHTNING =====
        # Logic này giúp tắt các thông báo của PyTorch Lightning
        try:
            import pytorch_lightning as pl
            pl.utilities.rank_zero._get_rank      = lambda: 1
            pl.utilities.rank_zero.rank_zero_only = lambda *a, **k: (lambda f: f)
            pl.utilities.rank_zero.rank_zero_info = lambda *a, **k: None
            pl.utilities.rank_zero.rank_zero_warn = lambda *a, **k: None
    
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    
            pl.loggers.base.LightningLoggerBase.info  = lambda *a, **k: None
            pl.loggers.base.LightningLoggerBase.warn  = lambda *a, **k: None
            pl.loggers.base.LightningLoggerBase.debug = lambda *a, **k: None
    
        except Exception:
            pass
        
        # ===== 1. BUILD HISTORY =====
        if self.freq is None:
            raise RuntimeError("Cần gọi fit() trước để thiết lập freq.")
    
        if self.last_y_train_ts is None or self.last_X_train_ts is None:
            raise RuntimeError("Thiếu history để dự báo.")
    
        y_hist = self.last_y_train_ts
        X_hist = self.last_X_train_ts
    
        if use_valid_history and self.last_y_valid_ts is not None:
            y_hist = y_hist.concatenate(self.last_y_valid_ts, axis=0)
            X_hist = X_hist.concatenate(self.last_X_valid_ts, axis=0)
    
        # ===== 2. CONVERT X → TimeSeries =====
        # Xử lý input (np.ndarray, DataFrame, TimeSeries) và chuyển thành TimeSeries (X_test_ts)
        if isinstance(X, np.ndarray):
            if self.feature_cols is None:
                raise RuntimeError("self.feature_cols chưa được thiết lập. Run fit() first.")
            X_df = pd.DataFrame(X, columns=self.feature_cols)
            # Gán index thời gian bắt đầu từ điểm cuối của lịch sử + 1
            start_date = y_hist.end_time() + pd.Timedelta(1, unit=self.freq)
            X_df.index = pd.date_range(start   = start_date, 
                                       periods = len(X_df), 
                                       freq    = self.freq)
            X_test_ts = TimeSeries.from_dataframe(X_df, freq=self.freq)
    
        elif isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.DatetimeIndex):
                start_date = y_hist.end_time() + pd.Timedelta(1, unit=self.freq)
                X.index = pd.date_range(start   = start_date, 
                                        periods = len(X), 
                                        freq    = self.freq)
            X_test_ts = TimeSeries.from_dataframe(X, freq=self.freq)
    
        elif isinstance(X, TimeSeries):
            X_test_ts = X
    
        else:
            raise TypeError("X phải là numpy array, DataFrame hoặc TimeSeries.")
    
        # ===== 3. SETUP TRAINER & ACCELERATOR LOGIC =====
        preds = []
        input_len = self.get_params()["input_chunk_length"]
    
        from pytorch_lightning import Trainer
    
        # ===== 3. SETUP TRAINER =====
        device = torch.cuda.is_available()
        trainer = Trainer(accelerator         = 'gpu' if device else 'cpu',
                          devices             = torch.cuda.device_count() if device else 1,
                          precision           = "64-true",
                          enable_progress_bar = False,
                          logger              = False)
    
        # ===== 4. PREDICTION LOGIC =====
        n = len(X_test_ts)
    
        if not rolling:
            # --- Dự đoán cả batch (Fast Prediction) ---
            # Chỉ lấy phần history covariates (cho encoder) cần thiết
            X_hist_part = X_hist.tail(input_len)

            # Ghép lịch sử X (cho encoder) + toàn bộ X_test_ts (cho decoder)
            X_full_ts = X_hist_part.concatenate(X_test_ts, axis=0)
            
            # Dự đoán toàn bộ n bước một lần
            y_pred_ts = self.model.predict(n               = n,
                                           series          = y_hist,
                                           past_covariates = X_full_ts,
                                           verbose         = False,
                                           n_jobs          = n_jobs,
                                           trainer         = trainer)
            
            # Lấy giá trị của target column đầu tiên
            preds = y_pred_ts.values()[:, 0] 
            
            if verbose:
                print(f"Dự báo batch {n} mẫu hoàn tất!")
            return np.array(preds)

        else:
            # --- Rolling forecast từng bước ---
            for i in range(n):
                if verbose and (i % max(1, n//10) == 0 or i == n-1):
                    sys.stdout.write(f"\rĐang dự báo cuốn chiếu: {i+1}/{n}")
                    sys.stdout.flush()
                    
                # Lấy 1 timestep của future covariates cần dự báo
                Xi = X_test_ts[i:i+1]
                
                # Lấy phần history covariates (cho encoder) cần thiết
                X_hist_part = X_hist.tail(input_len)
                
                # Future covariates cho Darts predict phải bao gồm cả phần history cho encoder và điểm dự báo
                X_full_ts = X_hist_part.concatenate(Xi, axis=0)
        
                # Thực hiện dự đoán 1 bước (n=1)
                y_pred_step = self.model.predict(n               = 1,
                                                 series          = y_hist, # Lịch sử output (y)
                                                 past_covariates = X_full_ts, # Lịch sử X + điểm X hiện tại
                                                 n_jobs          = n_jobs, 
                                                 verbose         = False,
                                                 trainer         = trainer)
                preds.append(y_pred_step.values()[0][0])
        
                # Cập nhật lịch sử (cuốn chiếu)
                y_hist = y_hist.concatenate(y_pred_step, axis=0)
                X_hist = X_hist.concatenate(Xi, axis=0)
        
            if verbose: print("\n[Hoàn tất dự báo]")
            return np.array(preds)



    def predict_history(self, type="train", verbose=True):
        """
        Tái tạo dự báo trên dữ liệu lịch sử (Train hoặc Valid).
        🚨 Logic: Nếu type='valid', sẽ nối thêm phần icl cuối của Train để dự báo được các điểm đầu Valid.
        """
        icl = self.model.input_chunk_length

        if type == "train":
            y_hist_ts = self.last_y_train_ts
            X_hist_ts = self.last_X_train_ts
            start_point = icl # Bắt đầu sau icl điểm đầu của train
        
        elif type == "valid":
            if self.last_y_valid_ts is None or self.last_y_train_ts is None:
                print(f"Cảnh báo: Thiếu dữ liệu Train hoặc Valid để dự báo history cho Valid.")
                return None
            
            # 🚨 KỸ THUẬT NỐI CHUỖI: Lấy icl điểm cuối của Train + toàn bộ Valid
            y_hist_ts = self.last_y_train_ts.tail(icl).concatenate(self.last_y_valid_ts, axis=0)
            X_hist_ts = self.last_X_train_ts.tail(icl).concatenate(self.last_X_valid_ts, axis=0)
            
            # Start tại icl giúp model bắt đầu dự báo ngay tại điểm đầu tiên của tập Valid gốc
            start_point = icl 
        
        else:
            raise ValueError("type phải là 'train' hoặc 'valid'.")

        if y_hist_ts is None or X_hist_ts is None:
            print(f"Cảnh báo: Không tìm thấy dữ liệu lịch sử cho type='{type}'. Bỏ qua predict_history.")
            return None
        
        # 2. Chạy historical_forecasts
        forecasts_ts_list = self.model.historical_forecasts(
            series           = y_hist_ts,
            past_covariates  = X_hist_ts, 
            start            = start_point,
            forecast_horizon = self.model.output_chunk_length, 
            retrain          = False,
            last_points_only = True,
            verbose          = verbose
        )
        
        # historical_forecasts trả về TimeSeries hoặc List[TimeSeries]
        if isinstance(forecasts_ts_list, TimeSeries):
            return forecasts_ts_list.values().flatten()
        else:
            # Darts 0.28.0+ cho phép .values().flatten() trực tiếp sau khi concatenate
            return TimeSeries.concatenate(forecasts_ts_list, axis=0).values().flatten()