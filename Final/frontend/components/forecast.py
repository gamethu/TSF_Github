import streamlit as st
import httpx
import pandas as pd
import joblib
import plotly.express as px
import sys
import re
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tft_model import TFTModel

# Make Drafts/Temp Prediction importable regardless of CWD or spaces in path
repo_root = Path(__file__).resolve().parents[3]
drafts_temp_pred = repo_root / "Drafts" / "Temp Prediction"
sys.path.insert(0, str(drafts_temp_pred))
from src.utilities import dataset

@st.cache_resource
def _infer_station_from_filename(data_file, station_options):
    filename = getattr(data_file, "name", "") or ""
    if not filename:
        return None

    stem = Path(filename).stem.lower()
    for station in station_options:
        token = station.lower()
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        if re.search(pattern, stem):
            return station

    return None

@st.cache_resource
def prepare_model_list(models, station, all_models):
    
    model_list = {}
    for m in models:
        model_list[m] = {"model"  : all_models[m]["model"][station],
                         "scaler" : all_models[m]["scaler"][station]}
        if "wrapper" in all_models[m]:
            model_list[m]["wrapper"] = all_models[m]["wrapper"][station]

    return model_list

@st.cache_resource
def prepare_df(df, target, station, all_stations, start=None, end=None):
    df_predict  = df.copy()
    df_predict  = dataset.fill_time_gaps(data       = df_predict,
                                         start_time = start,
                                         end_time   = end,
                                         freq       = "1D").reset_index(drop=True)

    df_predict  = dataset.HandleMissing_interpolate(data   = df_predict.set_index("time"),
                                                    method = "time")
    
    
    if station not in all_stations:
        st.warning(f"Station data for '{station}' not found.")
        return

    df_actual         = pd.read_csv(all_stations[station])
    df_actual["time"] = pd.to_datetime(df_actual["time"])
    df_actual         = df_actual.set_index("time")

    if target not in df_actual.columns:
        st.warning(f"Target '{target}' not found in station data.")
        return

    df_actual = df_actual.loc[start:end, target]

    return df_predict, df_actual

@st.cache_resource
def calculate_metrics(df_actual, res_df):
    common_index = df_actual.index.intersection(res_df.index)
    metrics_rows = []
    if not common_index.empty:
        actual_values = df_actual.loc[common_index]
        for model_name in res_df.columns:
            pred_values = res_df.loc[common_index, model_name]
            aligned = pd.concat([actual_values, pred_values], axis=1).dropna()
            if aligned.empty:
                continue

            y_true = aligned.iloc[:, 0]
            y_pred = aligned.iloc[:, 1]
            metrics_rows.append(
                {
                    "Model": model_name,
                    "R2": round(
                        r2_score(
                            y_true=y_true,
                            y_pred=y_pred,
                            multioutput="uniform_average",
                        )
                        * 100,
                        4,
                    ),
                    "MAE": round(
                        mean_absolute_error(
                            y_true=y_true,
                            y_pred=y_pred,
                            multioutput="uniform_average",
                        ),
                        4,
                    ),
                    "MSE": round(
                        mean_squared_error(
                            y_true=y_true,
                            y_pred=y_pred,
                            multioutput="uniform_average",
                        ),
                        4,
                    ),
                    "RMSE": round(
                        root_mean_squared_error(
                            y_true=y_true,
                            y_pred=y_pred,
                            multioutput="uniform_average",
                        ),
                        4,
                    ),
                    "MAPE": round(
                        mean_absolute_percentage_error(
                            y_true=y_true,
                            y_pred=y_pred,
                            multioutput="uniform_average",
                        ),
                        4,
                    ),
                }
            )
    return pd.DataFrame(metrics_rows)

@st.cache_resource
def predict_result(df_predict, model_list, start, end):
    res = {}

    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)

    hist_start = start - pd.Timedelta(days=7)
    pred_df    = df_predict.loc[hist_start:end]

    for name, model_pack in model_list.items():
        scaler_x = joblib.load(model_pack["scaler"]["x"])
        scaler_y = joblib.load(model_pack["scaler"]["y"])

        if "wrapper" in model_pack:
            model      = joblib.load(model_pack["wrapper"])
            model_path = str(model_pack["model"])

            if "NBEATS" in model_path:
                model.model = NBEATSModel.load(model_path, weights_only=False)
            elif "TFT" in model_path:
                model.model = TFTModel.load(model_path, weights_only=False)
            elif "TRANSFORMER" in model_path:
                model.model = TransformerModel.load(model_path, weights_only=False)
        else:
            model = joblib.load(model_pack["model"])

        X         = scaler_x.transform(pred_df)
        y_pred    = model.predict(X).reshape(-1, 1)

        res[name] = scaler_y.inverse_transform(y_pred).squeeze()

    return pd.DataFrame(res, index=pred_df.index).loc[start:end]

def gen_overview(overview_rows):
    if not overview_rows:
        return

    overview_df = pd.DataFrame(overview_rows)
    metric_cols = [col for col in overview_df.columns if col not in {"Station", "Model"}]

    model_overview_df = (
        overview_df
        .groupby("Model", as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    st.markdown("### Metrics overview by model")
    st.dataframe(model_overview_df.round(4), width='stretch')

    row_one = st.columns(3)
    row_two = st.columns(2)
    metric_layout = [
        (row_one[0], "R2"),
        (row_one[1], "MAE"),
        (row_one[2], "MSE"),
        (row_two[0], "RMSE"),
        (row_two[1], "MAPE"),
    ]

    for column, metric_name in metric_layout:
        if metric_name not in overview_df.columns:
            continue

        with column:
            fig = px.line(
                overview_df,
                x="Station",
                y=metric_name,
                color="Model",
                markers=True,
            )
            fig.update_layout(
                title=metric_name,
                xaxis_title="Station",
                yaxis_title=metric_name,
                hovermode="x unified",
                legend_title_text="Model",
            )
            st.plotly_chart(fig, width='stretch')

@st.cache_resource
def gen_result(df_actual, res_df, metrics_df, station):
    df_actual = df_actual.rename("Actual")

    with st.expander(label=f"{station} - Predicted Value", expanded=False):
        res_table = pd.concat([df_actual, res_df], axis=1)
        st.dataframe(res_table.round(2))

    with st.expander(label=f"{station} - Metrics", expanded=False):
        if not metrics_df.empty:
            st.dataframe(metrics_df)
        else:
            st.warning("No overlapping data to compute metrics.")

    with st.expander(label=f"{station} - Prediction Plot", expanded=False):
        plot_df = pd.concat([df_actual, res_df], axis=1)

        fig = px.line(
            plot_df,
            x=plot_df.index,
            y=plot_df.columns,
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted Value",
            hovermode="x unified",
            legend_title_text="Series",
        )

        st.plotly_chart(fig, width='stretch')

def process(files, targets, models):
    station_results = []
    overview_rows = []

    for file in files:
        if file is None:
            st.warning("Please upload a CSV file.")
            return

        if not targets:
            st.warning("Please choose a target.")
            return
        
        if isinstance(file, (str, Path)):
            df = pd.read_csv(file)
        else:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file)

        station_options = httpx.get("http://127.0.0.1:8000/stations/name").json()
        all_models      = httpx.get("http://127.0.0.1:8000/models/all").json()
        all_stations    = httpx.get("http://127.0.0.1:8000/stations/all").json()
        
        target     = targets[0]
        df["time"] = pd.to_datetime(df["time"])
        df         = df.set_index("time")
        start, end = df.index.min(), df.index.max()

        station = _infer_station_from_filename(file, station_options)
        if station is None:
            st.warning("Could not infer station from filename.")
            continue

        model_list = prepare_model_list(models, station, all_models)
        df_predict, df_actual = prepare_df(df, target, station, all_stations, start, end)

        if df_predict is None or df_actual is None:
            continue

        res_df = predict_result(df_predict, model_list, start, end)
        metrics_df = calculate_metrics(df_actual, res_df)

        if not metrics_df.empty:
            overview_rows.extend(
                {
                    "Station": station,
                    "Model": row["Model"],
                    "R2": row["R2"],
                    "MAE": row["MAE"],
                    "MSE": row["MSE"],
                    "RMSE": row["RMSE"],
                    "MAPE": row["MAPE"],
                }
                for _, row in metrics_df.iterrows()
            )

        station_results.append(
            {
                "station": station,
                "df_actual": df_actual,
                "res_df": res_df,
                "metrics_df": metrics_df,
            }
        )

    gen_overview(overview_rows)

    for result in station_results:
        with st.expander(label=result["station"], expanded=False):
            gen_result(
                result["df_actual"],
                result["res_df"],
                result["metrics_df"],
                result["station"],
            )
    