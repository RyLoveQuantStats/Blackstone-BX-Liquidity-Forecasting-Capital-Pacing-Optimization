"""
Simple Liquidity Forecasting Script
-----------------------------------
1. Loads master_data (capital_call_proxy) from SQLite.
2. Loads exogenous data (10Y Treasury Yield) if present.
3. Optionally differencing if the series is non-stationary.
4. Fits a fixed-order SARIMAX and an Exponential Smoothing model.
5. Averages both forecasts into an ensemble.
6. Outputs forecast, error metrics, and basic plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# Import DB and logging utilities
from utils.db_utils import get_connection, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error

# Setup logging
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

TABLE_NAME = "synthetic_master_data"
log_info(f"Final DB_PATH = {DB_PATH}")

# --------------------------------------------------------------------------
# 1. Data Loading
# --------------------------------------------------------------------------

def load_exogenous_data(db_path=DB_PATH):
    """
    Load a single exogenous column (10Y Treasury Yield) from macroeconomic_data.
    If it fails or column not found, return None.
    """
    try:
        conn = get_connection()
        # We assume there's a column named 'index' we rename to 'Date'.
        df_exo = pd.read_sql(
            'SELECT "index" as Date, "10Y Treasury Yield" FROM macroeconomic_data',
            conn, parse_dates=["Date"]
        )
        conn.close()
        df_exo.set_index("Date", inplace=True)
        df_exo.sort_index(inplace=True)
        if pd.infer_freq(df_exo.index) is None:
            df_exo = df_exo.asfreq("D")
        log_info("Exogenous data loaded successfully.")
        return df_exo
    except Exception as e:
        log_error(f"Error loading exogenous data: {e}")
        return None

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    """
    Loads the main data (capital_call_proxy) from the given table:
      - Sets Date as the index
      - Infers frequency or sets daily
      - Merges exogenous data if available
    """
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
        conn.close()
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        freq = pd.infer_freq(df.index)
        if not freq:
            df = df.asfreq("D")
            log_info("No frequency inferred; defaulting to daily.")
        if "capital_call_proxy" not in df.columns:
            return None

        # Merge exogenous data if available
        exo = load_exogenous_data(db_path)
        if exo is not None:
            df = df.merge(exo, left_index=True, right_index=True, how="left")
            log_info("Merged exogenous data.")

        return df
    except Exception as e:
        log_error(f"Error loading master data: {e}")
        return None

# --------------------------------------------------------------------------
# 2. Stationarity & Simple Checks
# --------------------------------------------------------------------------

def is_stationary(series, alpha=0.05):
    """
    Checks if series is stationary using the ADF test.
    """
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        log_info(f"ADF p-value = {p_value}")
        return p_value < alpha
    except Exception as e:
        log_error(f"ADF test failed: {e}")
        return False

# --------------------------------------------------------------------------
# 3. Forecasting Helpers
# --------------------------------------------------------------------------

def calculate_errors(actual, forecast):
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast)**2))
    mape = np.mean(np.abs((actual - forecast) / np.where(actual == 0, 1e-6, actual))) * 100
    return mae, rmse, mape

def plot_forecast(train, test, forecast, conf_int):
    """
    Plots train, test, forecast, and confidence intervals.
    """
    plt.figure(figsize=(10,5))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(forecast, label="Forecast", linestyle="--")
    if conf_int is not None:
        plt.fill_between(forecast.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                         color="gray", alpha=0.3, label="CI")
    plt.title("Capital Calls Forecast")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/capital_calls_forecast.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_residuals(resid):
    """
    Plots residuals, ACF, PACF, QQ-plot.
    """
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.plot(resid)
    plt.title("Residuals")

    plt.subplot(222)
    qqplot(resid, line="s", ax=plt.gca())
    plt.title("Q-Q Plot")

    plt.subplot(223)
    plot_acf(resid.dropna(), ax=plt.gca())
    plt.title("ACF")

    plt.subplot(224)
    plot_pacf(resid.dropna(), ax=plt.gca(), method="ywm")
    plt.title("PACF")

    plt.tight_layout()
    save_path = "plots/residuals.png"
    plt.savefig(save_path)
    plt.close()

    lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
    pvalue = lb["lb_pvalue"].values[0]
    return save_path, pvalue

# --------------------------------------------------------------------------
# 4. Main Forecasting Function
# --------------------------------------------------------------------------

def main():
    df = load_master_data()
    if df is None or "capital_call_proxy" not in df.columns:
        return {"error": "capital_call_proxy column not found or data load failed."}

    # Keep only rows with non-null capital_calls
    df = df[df["capital_call_proxy"].notnull()]

    # If the series is constant or too small, return a trivial forecast
    series = df["capital_call_proxy"]
    if series.nunique() <= 1:
        log_error("capital_calls is constant. Returning trivial forecast.")
        return {
            "error": "capital_calls is constant or insufficient data.",
            "trivial_forecast": float(series.iloc[-1]) if len(series) > 0 else 0.0
        }

    # Optional differencing if non-stationary
    if not is_stationary(series):
        log_info("Series not stationary; differencing once.")
        series = series.diff().dropna()

    # Split train/test (80/20)
    train_size = int(len(series)*0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    # If exogenous present, use it
    if "10Y Treasury Yield" in df.columns:
        exog = df["10Y Treasury Yield"]
        train_exog = exog.iloc[:train_size]
        test_exog = exog.iloc[train_size:]
    else:
        train_exog = test_exog = None

    # Simple fixed orders for SARIMAX
    order = (1,1,1)
    seasonal_order = (1,0,1,7)

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    exog=train_exog, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Forecast
    steps = len(test)
    forecast_obj = results.get_forecast(steps=steps, exog=test_exog)
    forecast_sarimax = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    forecast_sarimax.index = test.index
    conf_int.index = test.index

    # Fit Exponential Smoothing
    es_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=7).fit()
    forecast_es = es_model.forecast(steps)
    forecast_es.index = test.index

    # Simple average ensemble
    forecast_ensemble = (forecast_sarimax + forecast_es) / 2

    # Metrics
    mae, rmse, mape = calculate_errors(test.values, forecast_ensemble.values)
    log_info(f"MAE={mae}, RMSE={rmse}, MAPE={mape}")

    # Plots
    forecast_plot = plot_forecast(train, test, forecast_ensemble, conf_int)
    residuals_plot, lb_pvalue = plot_residuals(results.resid.dropna())

    output = {
        "model_summary": results.summary().as_text(),
        "error_metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape},
        "forecast": forecast_ensemble.to_dict(),
        "confidence_intervals": conf_int.to_dict(),
        "forecast_plot": forecast_plot,
        "residuals_plot": residuals_plot,
        "ljung_box_pvalue": lb_pvalue,
        "model_order": {"order": order, "seasonal_order": seasonal_order},
        "transformation": "diff once if non-stationary"
    }
    return output

def run():
    return main()

if __name__ == "__main__":
    result = run()
    # Convert timestamps to strings for JSON
    if isinstance(result.get("forecast"), dict):
        result["forecast"] = {str(k): v for k, v in result["forecast"].items()}
    if isinstance(result.get("confidence_intervals"), dict):
        for bound in result["confidence_intervals"]:
            result["confidence_intervals"][bound] = {
                str(k): v for k, v in result["confidence_intervals"][bound].items()
            }
    print(json.dumps(result, indent=4))
