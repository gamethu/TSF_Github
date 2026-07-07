import streamlit as st
import httpx
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
import joblib
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tft_model import TFTModel

src_name = dict({"CaMau" : "Cà Mau",
                 "DH"    : "Đồng Hới",
                 "NB"    : "Nội Bài",
                 "QN"    : "Quy Nhơn",
                 "TH"    : "Thanh Hóa",
                 "TSN"   : "Tân Sơn Nhất"})

# Tần suất resample cho từng chu kỳ -> mỗi step là 1 mốc thời gian tuần tự
# (vd: 2 năm theo tháng = 24 step, 2 năm theo quý = 8 step)
STEP_FREQ  = {"quarter": "Q", "month": "M"}
STEP_TITLE = {"quarter": "Quarter", "month": "Month"}

# Thứ tự và nhãn hiển thị của các metric (mỗi metric là 1 plot riêng)
METRIC_ORDER = ["R2", "MAE", "MSE", "RMSE", "MAPE"]
METRIC_TITLE = {"R2": "R2 (%)", "MAE": "MAE", "MSE": "MSE", "RMSE": "RMSE", "MAPE": "MAPE"}

@st.cache_resource
def gen_model_data(stations, model_data):
    reg_model = dict()
    scaler_x  = dict()
    scaler_y  = dict()

    for name in stations:
        if "ML" in model_data["model"][name]:
            reg_model[name] = joblib.load(model_data["model"][name])
        else: # DL
            # # Load lại wrapper
            reg_model[name] = joblib.load(model_data["wrapper"][name])
            
            # # Load lại model
            if "NBEATS" in model_data["model"][name]:
                reg_model[name].model = NBEATSModel.load(model_data["model"][name],weights_only=False)
            elif "TFT" in model_data["model"][name]:
                reg_model[name].model = TFTModel.load(model_data["model"][name],weights_only=False)
            elif "TRANSFORMER" in model_data["model"][name]:
                reg_model[name].model = TransformerModel.load(model_data["model"][name],weights_only=False)
        
        
        scaler_x[name] = joblib.load(model_data["scaler"][name]["x"])
        scaler_y[name] = joblib.load(model_data["scaler"][name]["y"])
    return reg_model, scaler_x, scaler_y

@st.cache_resource
def gen_model_output(stations, output_data):
    station_fit   = dict()
    station_valid = dict()
    station_pred  = dict()

    for name in stations:
        station_fit[name]       = pd.DataFrame(pd.read_csv(output_data[name][".csv"]["train"], index_col="time")["y_pred"].iloc[7:])
        station_fit[name].index = pd.to_datetime(station_fit[name].index)

        station_valid[name]       = pd.DataFrame(pd.read_csv(output_data[name][".csv"]["valid"], index_col="time")["y_pred"])
        station_valid[name].index = pd.to_datetime(station_valid[name].index)

        station_pred[name]       = pd.DataFrame(pd.read_csv(output_data[name][".csv"]["test"], index_col="time")["y_pred"])
        station_pred[name].index = pd.to_datetime(station_pred[name].index)
    return station_fit, station_valid, station_pred

def gen_feature_importance(station, model_data, feature):
    reg_model, _, _ = gen_model_data([station], model_data)
    model = reg_model[station]

    # Chỉ model ML mới có feature importance
    if not hasattr(model, "feature_importances_"):
        st.info("Model này không hỗ trợ feature importance.")
        return

    # Tạo dictionary feature importance
    fi = dict({k: v for k, v in sorted(zip(feature,
                                           model.feature_importances_),
                                       key     = lambda x: x[1],
                                       reverse = True)})

    fig, ax = plt.subplots(figsize            = (6, 4),
                           constrained_layout = True)
    sns.barplot(fi, orient='h', ax=ax)
    ax.grid(True, axis='x')
    ax.set_title(f"Mức độ quan trọng của đặc trưng — {src_name.get(station, station)}")
    st.pyplot(fig)

@st.cache_resource
def gen_evaluate_metrics(stations, model_data, output_data):
    station_fit, station_valid, station_pred = gen_model_output(stations, output_data)
    
    r2 = dict({"train" : dict(),
               "valid" : dict(),
               "test"  : dict()})
    mae = dict({"train" : dict(),
                "valid" : dict(),
                "test"  : dict()})
    mse = dict({"train" : dict(),
                "valid" : dict(),
                "test"  : dict()})
    mape = dict({"train" : dict(),
                 "valid" : dict(),
                 "test"  : dict()})
    rmse = dict({"train" : dict(),
                 "valid" : dict(),
                 "test"  : dict()})
    
    for name in stations:
        y_train = pd.read_csv(output_data[name][".csv"]["train"],index_col="time")
        y_valid = pd.read_csv(output_data[name][".csv"]["valid"],index_col="time")
        y_test  = pd.read_csv(output_data[name][".csv"]["test"],index_col="time")
        
        # R2
        r2["train"][name] = r2_score(y_true      = y_train[["y_true"]].iloc[7:],
                                     y_pred      = station_fit[name],
                                     multioutput = "uniform_average")
        r2["valid"][name] = r2_score(y_true      = y_valid[["y_true"]],
                                     y_pred      = station_valid[name],
                                     multioutput = "uniform_average")
        r2["test"][name] = r2_score(y_true      = y_test[["y_true"]],
                                    y_pred      = station_pred[name],
                                    multioutput = "uniform_average")
        
        # MAE
        mae["train"][name] = mean_absolute_error(y_true      = y_train[["y_true"]].iloc[7:],
                                                 y_pred      = station_fit[name],
                                                 multioutput = "uniform_average")
        mae["valid"][name] = mean_absolute_error(y_true      = y_valid[["y_true"]],
                                                 y_pred      = station_valid[name],
                                                 multioutput = "uniform_average")
        mae["test"][name] = mean_absolute_error(y_true      = y_test[["y_true"]],
                                                y_pred      = station_pred[name],
                                                multioutput = "uniform_average")
        
        # MSE
        mse["train"][name] = mean_squared_error(y_true      = y_train[["y_true"]].iloc[7:],
                                                y_pred      = station_fit[name],
                                                multioutput = "uniform_average")
        mse["valid"][name] = mean_squared_error(y_true      = y_valid[["y_true"]],
                                                y_pred      = station_valid[name],
                                                multioutput = "uniform_average")
        mse["test"][name] = mean_squared_error(y_true      = y_test[["y_true"]],
                                               y_pred      = station_pred[name],
                                               multioutput = "uniform_average")
        
        # RMSE
        rmse["train"][name] = root_mean_squared_error(y_true      = y_train[["y_true"]].iloc[7:],
                                                      y_pred      = station_fit[name],
                                                      multioutput = "uniform_average")
        rmse["valid"][name] = root_mean_squared_error(y_true      = y_valid[["y_true"]],
                                                      y_pred      = station_valid[name],
                                                      multioutput = "uniform_average")
        rmse["test"][name] = root_mean_squared_error(y_true      = y_test[["y_true"]],
                                                     y_pred      = station_pred[name],
                                                     multioutput = "uniform_average")
        
        # MAPE
        mape["train"][name] = mean_absolute_percentage_error(y_true      = y_train[["y_true"]].iloc[7:],
                                                             y_pred      = station_fit[name],
                                                             multioutput = "uniform_average")
        mape["valid"][name] = mean_absolute_percentage_error(y_true      = y_valid[["y_true"]],
                                                             y_pred      = station_valid[name],
                                                             multioutput = "uniform_average")
        mape["test"][name] = mean_absolute_percentage_error(y_true      = y_test[["y_true"]],
                                                             y_pred      = station_pred[name],
                                                             multioutput = "uniform_average")
    return r2, mae, mse, mape, rmse

def gen_metric_dataframe(stations, model_data, output_data, metrics):
    r2, mae, mse, mape, rmse = gen_evaluate_metrics(stations, model_data, output_data)
    
    metric_map = {
        "R2"   : r2,
        "MAE"  : mae,
        "MSE"  : mse,
        "MAPE" : mape,
        "RMSE" : rmse,
    }
    sets = ["train", "valid", "test"]
    data = list()
    
    for set in sets:
        for station in stations:
            row = {"set": set, "station": station}

            for metric_name in metrics:  # metrics = ["R2", "MAE", ...]
                values = metric_map[metric_name]

                if station in values[set]:
                    value = values[set][station]

                    if metric_name == "R2":
                        value = round(value * 100, 4)
                    else:
                        value = round(value, 4)

                    row[metric_name] = value

            data.append(row)

    st.dataframe(data)

def gen_actual_vs_predict(csv_path):
    df = pd.read_csv(csv_path, index_col="time")
    df.index = pd.to_datetime(df.index)

    cols = st.columns(2)
    with cols[0]:
        fig = px.line()

        fig.add_scatter(x    = df.index,
                        y    = df["y_true"],
                        mode = "lines",
                        name = "Thực tế",
                        line = dict(color="orange"))

        fig.add_scatter(x    = df.index,
                        y    = df["y_pred"],
                        mode = "lines",
                        name = "Dự đoán",
                        line = dict(color="black"))

        fig.update_layout(title       = "Actual vs Predict",
                          xaxis_title = "Time",
                          yaxis_title = "Value",
                          hovermode   = "x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"actual_vs_predict_line{csv_path}")
    with cols[1]:
        fig = px.scatter()

        # Scatter: Actual vs Predict
        fig.add_scatter(x    = df["y_true"],
                        y    = df["y_pred"],
                        mode = "markers",
                        marker = dict(color="blue"))

        # Đường y = x (perfect prediction)
        min_val = min(df["y_true"].min(), df["y_pred"].min())
        max_val = max(df["y_true"].max(), df["y_pred"].max())

        fig.add_scatter(x    = [min_val, max_val],
                        y    = [min_val, max_val],
                        mode = "lines",
                        line = dict(color="red", dash="dash"))

        fig.update_layout(title       = "Actual vs Predict",
                          xaxis_title = "Thực tế",
                          yaxis_title = "Dự đoán",
                          hovermode   = "closest")
        fig.update_xaxes(showspikes     = True,
                         spikecolor     = "grey",
                         spikethickness = 1,
                         spikedash      = "dot")

        fig.update_yaxes(showspikes     = True,
                         spikecolor     = "grey",
                         spikethickness = 1,
                         spikedash      = "dot")

        st.plotly_chart(fig, use_container_width=True, key=f"actual_vs_predict_scatter{csv_path}")

def gen_actual_vs_predict_sets(output_data, station):
    """Vẽ Actual vs Predict cho cả 3 tập train/valid/test của 1 trạm."""
    for split, label in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
        with st.expander(label=label, expanded=False):
            gen_actual_vs_predict(output_data[station][".csv"][split])

@st.cache_resource
def _step_metrics(station, output_data, cycle_type, metrics, split):
    """Tính metric theo từng step thời gian tuần tự cho 1 tập (train/valid/test).

    Mỗi step là 1 mốc thời gian (year-month hoặc year-quarter): vd 5 năm theo
    tháng -> 60 step, theo quý -> 20 step.
    """
    df = pd.read_csv(output_data[station][".csv"][split])
    if not {"time", "y_true", "y_pred"}.issubset(df.columns):
        return pd.DataFrame()

    df = df[["time", "y_true", "y_pred"]].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "y_true", "y_pred"])
    if df.empty:
        return pd.DataFrame()

    df["step"] = df["time"].dt.to_period(STEP_FREQ[cycle_type])

    rows = []
    for step_value, g in df.groupby("step", sort=True):
        y_true, y_pred = g["y_true"], g["y_pred"]
        row = {"step": str(step_value)}
        if "R2" in metrics:
            row["R2"]   = r2_score(y_true, y_pred, multioutput="uniform_average") * 100
        if "MAE" in metrics:
            row["MAE"]  = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
        if "MSE" in metrics:
            row["MSE"]  = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
        if "RMSE" in metrics:
            row["RMSE"] = root_mean_squared_error(y_true, y_pred, multioutput="uniform_average")
        if "MAPE" in metrics:
            row["MAPE"] = mean_absolute_percentage_error(y_true, y_pred, multioutput="uniform_average")
        rows.append(row)

    return pd.DataFrame(rows)

def _single_metric_chart(metric, per_model, color_of):
    fig = go.Figure()
    for model, table in per_model.items():
        if metric not in table.columns:
            continue
        fig.add_scatter(x=table["step"].astype(str),
                        y=table[metric],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=color_of[model]))
    fig.update_layout(title=METRIC_TITLE[metric], xaxis_title="Step",
                      yaxis_title=METRIC_TITLE[metric], hovermode="x unified")
    # Giữ thứ tự step theo thời gian (không để plotly sắp xếp lại)
    fig.update_xaxes(type="category")
    return fig

def gen_station_ranking(station, models, all_output, metrics, cycle_types, color_of):
    """5 hình so sánh model (mỗi metric 1 plot) cho 1 trạm, x là step thời gian.

    Chia theo từng tập train/valid/test (mỗi tập 1 expander) giống Actual vs
    Predict; trong mỗi tập lại chia theo chu kỳ (tháng/quý).
    """
    if not metrics or not cycle_types:
        return

    ordered_metrics = [m for m in METRIC_ORDER if m in metrics]

    for split, label in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
        with st.expander(label=label, expanded=False):
            for cycle_type in cycle_types:
                if cycle_type not in STEP_FREQ:
                    continue

                per_model = {}
                for model in models:
                    table = _step_metrics(station, all_output[model], cycle_type, metrics, split)
                    if not table.empty:
                        per_model[model] = table
                if not per_model:
                    continue

                # Mỗi chu kỳ (tháng / quý) là 1 expander con
                with st.expander(label=STEP_TITLE[cycle_type], expanded=False):
                    cols = st.columns(2)
                    for i, metric in enumerate(ordered_metrics):
                        with cols[i % 2]:
                            fig = _single_metric_chart(metric, per_model, color_of)
                            st.plotly_chart(fig, use_container_width=True, key=f"{station}_{split}_{cycle_type}_{metric}")

def process(models, stations, charts, metrics, feature, cycle_ranking=None):
    all_output = httpx.get("http://127.0.0.1:8000/stations/model_predict").json()
    all_model  = httpx.get("http://127.0.0.1:8000/models/all").json()

    cycle_label_map = {
        "Quarter": "quarter",
        "Month": "month",
    }
    selected_cycles = []
    for cycle in (cycle_ranking or []):
        if cycle in ["quarter", "month"]:
            selected_cycles.append(cycle)
        elif cycle in cycle_label_map:
            selected_cycles.append(cycle_label_map[cycle])

    palette  = px.colors.qualitative.Plotly
    color_of = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    # Mỗi khu vực (trạm) là 1 expander chứa các expander con
    for station in stations:
        with st.expander(label=src_name.get(station, station), expanded=False):

            # Metric -> expander theo model
            with st.expander(label="Metric", expanded=False):
                for model in models:
                    with st.expander(label=model, expanded=False):
                        gen_metric_dataframe([station], all_model[model], all_output[model], metrics)

            # Feature Importance -> expander theo model
            if "Feature Importance" in charts:
                with st.expander(label="Feature Importance", expanded=False):
                    for model in models:
                        with st.expander(label=model, expanded=False):
                            gen_feature_importance(station, all_model[model], feature)

            # Actual vs Predict -> expander theo model
            if "Actual vs Predict" in charts:
                with st.expander(label="Actual vs Predict", expanded=False):
                    for model in models:
                        with st.expander(label=model, expanded=False):
                            gen_actual_vs_predict_sets(all_output[model], station)

            # Cycle Ranking -> expander, trong đó mỗi chu kỳ (tháng/quý) 1 expander
            with st.expander(label="Cycle Ranking", expanded=False):
                gen_station_ranking(station, models, all_output, metrics,
                                    selected_cycles, color_of)