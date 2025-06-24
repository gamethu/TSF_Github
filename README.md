# Temperature Time Series Forecasting

Dự án dự báo nhiệt độ theo chuỗi thời gian (Time Series Forecasting) sử dụng các thuật toán Machine Learning và Deep Learning như Random Forest, XGBoost, LSTM, Transformer, Temporal Fusion Transformer (TFT), N-BEATS, hướng đến áp dụng thực tiễn tại các khu vực đô thị và ven biển Việt Nam.

## ✨ Thông tin dự án

* **Tên đề tài:** Ứng dụng mô hình học sâu dự đoán nhiệt độ cực đại hàng ngày tại các đô thị Việt Nam
* **Thời gian:** 08/2025 – 02/2026
* **Sinh viên thực hiện:** Nhóm 3 sinh viên khoa CNTT – Trường ĐH Sài Gòn
* **GVHD:** Cô Nguyễn Thị Tuyết Nam, Thầy Nguyễn Trung Tín

## 📂 Cấu trúc thư mục dự án

```
Temp Prediction/
├— data/
│   ├— raw/                # Dữ liệu gốc
│   └— processed/          # Dữ liệu đã xử lý
│
├— models/
│   ├— checkpoints/        # Model đang train
│   └— trained_models/     # Model đã train xong
│
├— notebooks/              # Notebook exploratory và EDA
│   └— *.ipynb
│
├— output/                 # Kết quả dự báo và log
│   └— forecast_results.csv
│
├— scripts/                # Script train/test/chạy model
│   ├— train_model.py
│   ├— run_forecast.py
│   └— evaluate_model.py
│
├— src/                    # Mã nguồn chính
│   ├— config.py
│   ├— dataset.py
│   ├— features.py
│   ├— plots.py
│   └— modeling/       # Thư viện mô hình
│       └— *.py (LSTM, N-BEATS...)
│
├— requirements.txt        # Danh sách thư viện
├— README.md               # Mô tả dự án
└— .gitignore
```

## 📆 Dataset

* **Nguồn dữ liệu:** ERA5 (ECMWF ReAnalysis v5) do trung tâm ECMWF cung cấp
* **Thông tin:** Dữ liệu nhiệt độ cực đại hàng ngày, độ che phủ mây, hướng gó, địa hình (kinh độ, vĩ độ, khu vực đô thị/ven biển).
* **Thời gian:** 1990 – 2024
* **Định dạng:** .csv hoặc grib trước khi tách

## 🚀 Mô hình áp dụng

### 📏 Machine Learning:

* Random Forest
* XGBoost

### 🔎 Deep Learning:

* LSTM (Long Short-Term Memory)
* Transformer
* N-BEATS
* TFT (Temporal Fusion Transformer)

## ⚙️ Cài đặt

Tạo môi trường ảo và cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## ✅ Cách chạy

### 1. Khám phá và xử lý dữ liệu:

* Chạy notebook trong `notebooks/`

### 2. Train model:

```bash
python scripts/train_model.py
```

### 3. Chạy dự báo:

```bash
python scripts/run_forecast.py
```

### 4. Đánh giá kết quả:

```bash
python scripts/evaluate_model.py
```

## 📊 Output

* Dự báo nhiệt độ: `output/forecast_results.csv`
* Model checkpoints: `models/checkpoints/`
* Model train xong: `models/trained_models/`

## 📖 Tài liệu tham khảo

* ERA5, ECMWF
* Nghiên cứu TFT, N-BEATS, LSTM forecasting
* Tổng quan khoa học trong/ngoài nước (chi tiết trong file PDF thuyết minh)

## 🗓️ Tiến độ thực hiện (08/2025 – 02/2026)

* T8-9: Thu thập dữ liệu, tiền xử lý
* T10-11: Phát triển model
* T12-1: Thực nghiệm & đánh giá
* T2: Hoàn thiện báo cáo

## 💼 Bản quyền & Liên hệ

* Nhóm NCKH SV – Trường ĐH Sài Gòn
* Email: [nguyentandai.nckh@gmail.com](mailto:nguyentandai.nckh@gmail.com)
* Dự án mang tính học thuật phi lợi nhuận. Vui lòng ghi rõ nguồn khi sử dụng.

---

> ✔️ Dự án đã tích hợp: Random Forest, XGBoost, LSTM, Transformer, TFT, N-BEATS ✔️ Tối ưu chạy được trên laptop hoặc GPU NVIDIA qua Google Colab