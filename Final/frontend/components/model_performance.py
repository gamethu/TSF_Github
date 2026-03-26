import streamlit as st
import httpx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
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
            print(model_data["wrapper"][name])
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

@st.cache_resource
def gen_feature_importance(stations, model_data, feature):
    reg_model, _, _ = gen_model_data(stations, model_data)
    
    num_cols = len(stations)

    # Xác định số hàng và số cột hợp lý
    ncols = 2  # Số biểu đồ trên mỗi hàng
    nrows = int(np.ceil(num_cols / ncols))  # Tính số hàng cần thiết

    fig, ax = plt.subplots(ncols              = ncols, 
                           nrows              = nrows, 
                           figsize            = (5*ncols, 4*nrows),
                           constrained_layout = True)  # auto căn chỉnh
    ax = ax.flatten()  # chuyển mảng 2 chiều thành 1 chiều để dễ duyệt

    count = list(["a", "b", "c", "d", "e", "f"])
    for i, name in enumerate(stations, 0):
        
        # Tạo dictionary feature importance
        fi = dict({k: v for k,v in sorted(zip(feature,
                                              reg_model[name].feature_importances_),
                                          key     = lambda x : x[1],
                                          reverse = True)})
        
        # Plot
        sns.barplot(fi, orient='h', ax=ax[i])
        ax[i].grid(True, axis='x')
        ax[i].set_title(f"({count[i]}) {src_name[name]}")

    fig.suptitle("Mức độ quan trọng của đặc trưng tại các trạm khí tượng", 
                fontsize   = 15, 
                fontweight = 'bold', 
                ha         = 'center')
    fig.tight_layout()
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

@st.cache_resource
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

@st.cache_resource
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
        st.plotly_chart(fig, use_container_width=True)
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

        st.plotly_chart(fig, use_container_width=True)
                
def gen_summary(stations, output_data, avp):
    for name in stations:
        with st.expander(label    = name,
                         expanded = True):
            with st.expander(label    = "Train",
                             expanded = False):
                if avp:
                    gen_actual_vs_predict(output_data[name][".csv"]["train"])
            with st.expander(label    = "Valid",
                             expanded = False):
                if avp:
                    gen_actual_vs_predict(output_data[name][".csv"]["valid"])
            with st.expander(label    = "Test",
                             expanded = False):
                if avp:
                    gen_actual_vs_predict(output_data[name][".csv"]["test"])

def _metric_orientation(metric_name):
    return False if metric_name == "R2" else True

def gen_periodic_metric_ranking(stations, output_data, metrics, cycle_types=None):
    if not metrics:
        st.info("Chưa chọn metrics để ranking.")
        return

    if not cycle_types:
        st.info("Chưa chọn chu kỳ thời gian để ranking.")
        return

    rows = []

    for station in stations:
        test_path = output_data[station][".csv"]["test"]
        df = pd.read_csv(test_path)

        if not {"time", "y_true", "y_pred"}.issubset(df.columns):
            continue

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "y_true", "y_pred"]).copy()
        if df.empty:
            continue

        df["weekday"] = df["time"].dt.day_name()
        df["quarter"] = "Q" + df["time"].dt.quarter.astype(str)
        df["month"]   = df["time"].dt.month
        print(df)
        cycle_map = {
            "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "quarter": ["Q1", "Q2", "Q3", "Q4"],
            "month": list(range(1, 13)),
        }

        for cycle_type in cycle_types:
            if cycle_type not in cycle_map:
                continue

            ordered_values = cycle_map[cycle_type]
            for cycle_value in ordered_values:
                g = df[df[cycle_type] == cycle_value]
                if g.empty:
                    continue

                row = {
                    "cycle_type": cycle_type,
                    "cycle": cycle_value,
                    "station": station,
                    "samples": len(g),
                }

                y_true = g["y_true"]
                y_pred = g["y_pred"]

                if "R2" in metrics:
                    row["R2"] = r2_score(y_true=y_true, y_pred=y_pred, multioutput = "uniform_average") * 100
                if "MAE" in metrics:
                    row["MAE"] = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput = "uniform_average")
                if "MSE" in metrics:
                    row["MSE"] = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput = "uniform_average")
                if "RMSE" in metrics:
                    row["RMSE"] = root_mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput = "uniform_average")
                if "MAPE" in metrics:
                    row["MAPE"] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput = "uniform_average")

                rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        st.info("Không đủ dữ liệu để tạo ranking theo chu kỳ đã chọn.")
        return

    ranking["cycle"] = ranking.apply(
        lambda r: f"M{int(r['cycle']):02d}" if r["cycle_type"] == "month" else r["cycle"],
        axis=1,
    )

    rank_cols = []
    for metric_name in metrics:
        if metric_name not in ranking.columns:
            continue

        ascending = _metric_orientation(metric_name)
        rank_col = f"rank_{metric_name}"
        ranking[rank_col] = ranking.groupby(["cycle_type", "cycle"])[metric_name].rank(
            ascending=ascending,
            method="min"
        )
        rank_cols.append(rank_col)

    if rank_cols:
        ranking["rank_score"] = ranking[rank_cols].mean(axis=1)
        ranking["rank_overall"] = ranking.groupby(["cycle_type", "cycle"])["rank_score"].rank(
            ascending=True,
            method="min",
        )
    else:
        ranking["rank_score"] = np.nan
        ranking["rank_overall"] = np.nan

    for metric_name in metrics:
        if metric_name in ranking.columns:
            digits = 4 if metric_name != "R2" else 4
            ranking[metric_name] = ranking[metric_name].round(digits)

    if "rank_score" in ranking.columns:
        ranking["rank_score"] = ranking["rank_score"].round(3)

    for col in ["samples", "rank_overall"] + rank_cols:
        if col in ranking.columns:
            ranking[col] = ranking[col].astype("Int64")

    ranking = ranking.sort_values(["cycle_type", "cycle", "rank_overall", "station"])
    st.dataframe(ranking)
            
def process(model, stations, charts, metrics, feature, cycle_ranking=None):
    output_data = httpx.get("http://127.0.0.1:8000/stations/model_predict").json()[model]
    model_data  = httpx.get("http://127.0.0.1:8000/models/all").json()[model]

    cycle_label_map = {
        "Theo quý": "quarter",
        "Theo thứ": "weekday",
        "Theo tháng": "month",
    }
    selected_cycles = []
    for cycle in (cycle_ranking or []):
        if cycle in ["quarter", "weekday", "month"]:
            selected_cycles.append(cycle)
        elif cycle in cycle_label_map:
            selected_cycles.append(cycle_label_map[cycle])
    
    gen_metric_dataframe(stations, model_data, output_data, metrics)

    gen_periodic_metric_ranking(stations, output_data, metrics, selected_cycles)
    
    if "Feature Importance" in charts:
        gen_feature_importance(stations, model_data, feature)
    
    with st.container(border=False):
        with st.expander(label    = "Summary",
                         expanded = True):
            gen_summary(stations, output_data, 
                        avp = True if "Actual vs Predict" in charts else None)