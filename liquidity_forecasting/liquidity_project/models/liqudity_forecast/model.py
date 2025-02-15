"""
Liquidity Forecasting: SARIMAX Model with Macroeconomic Variables
-----------------------------------------------------------------
This script loads the merged KKR dataset from a centralized SQLite database,
performs SARIMAX forecasting on the 'capital_calls' time series (with exogenous variables),
and outputs:
  - A summary of the fitted SARIMAX model and its parameters.
  - Forecast error metrics (MAE, RMSE, MAPE).
  - Forecast and residual diagnostic plots (with ACF, PACF, QQ plot, and histogram; with shaded confidence intervals).
  - Seasonal and nonseasonal model orders selected via grid search.
  - An ensemble forecast that averages SARIMAX and Exponential Smoothing forecasts.
  - JSON output ready for integration (Django API).
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import json
import logging
from scipy import stats

# Import centralized DB and logging utilities.
from utils.db_utils import get_connection, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error

# Set up logging (both file and console).
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

TABLE_NAME = "master_data"
log_info(f"Final DB_PATH = {DB_PATH}")

def load_exogenous_data(db_path=DB_PATH):
    """
    Load exogenous variables from the 'macroeconomic_data' table.
    Here we assume a column named "10Y Treasury Yield" exists.
    """
    try:
        conn = get_connection()
        df_exo = pd.read_sql("SELECT Date, `10Y Treasury Yield` FROM macroeconomic_data", conn, parse_dates=["Date"])
        conn.close()
        df_exo.set_index("Date", inplace=True)
        df_exo.sort_index(inplace=True)
        inferred_freq = pd.infer_freq(df_exo.index)
        if not inferred_freq:
            df_exo = df_exo.asfreq('D')
        log_info("Exogenous data loaded successfully.")
        return df_exo
    except Exception as e:
        log_error(f"Error loading exogenous data: {e}")
        return None

def remove_outliers_mad(series, threshold=3.5):
    """
    Remove outliers from a pandas Series using the Median Absolute Deviation (MAD) method.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z = 0.6745 * (series - median) / (mad + 1e-6)
    return series[np.abs(modified_z) < threshold]

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    """
    Load the master dataset from the SQLite database.
    - Sets 'Date' as the index.
    - Attempts to infer frequency; if none, sets frequency to daily ('D').
    - Drops rows with missing 'capital_calls' and removes outliers using MAD.
    - Merges in exogenous data if available.
    """
    log_info(f"Using DB_PATH = {db_path}")
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
        conn.close()
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq:
            df.index.freq = inferred_freq
            log_info(f"Inferred frequency: {inferred_freq}")
        else:
            log_info("Could not infer frequency from Date index. Setting frequency to 'D' (daily).")
            df = df.asfreq('D')
        df = df[df["capital_calls"].notnull()]
        # Remove outliers using MAD.
        df["capital_calls"] = remove_outliers_mad(df["capital_calls"])
        log_info("Outlier removal complete.")
        # Merge exogenous data if available.
        df_exo = load_exogenous_data(db_path)
        if df_exo is not None:
            df = df.merge(df_exo, left_index=True, right_index=True, how="left")
            log_info("Merged exogenous data.")
        return df
    except Exception as e:
        log_error(f"Error loading data: {e}")
        raise

def select_sarimax_order(ts, exog=None,
                         p_values=[0, 1, 2], d_values=[1], q_values=[0, 1, 2],
                         seasonal_orders=[(0, 0, 0, 7), (1, 0, 1, 7)]):
    """
    Grid search over nonseasonal (p,d,q) and seasonal (P,D,Q,s) orders
    to select the best SARIMAX model based on AIC.
    Optionally uses exogenous data.
    Returns (order, seasonal_order).
    """
    best_aic = np.inf
    best_order = None
    best_seasonal = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for seasonal_order in seasonal_orders:
                    try:
                        model = SARIMAX(ts,
                                        order=(p, d, q),
                                        seasonal_order=seasonal_order,
                                        exog=exog,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                        results = model.fit(disp=False)
                        log_info(f"Tested order ({p},{d},{q}) seasonal {seasonal_order} AIC: {results.aic}")
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order
                    except Exception as e:
                        log_info(f"WARNING: Order ({p},{d},{q}) seasonal {seasonal_order} failed: {e}")
    if best_order is None or best_seasonal is None:
        raise ValueError("No suitable SARIMAX model found.")
    log_info(f"Selected order {best_order} and seasonal_order {best_seasonal} with AIC: {best_aic}")
    return best_order, best_seasonal

def calculate_error_metrics(actual, forecast):
    """
    Calculate MAE, RMSE, and MAPE (with safeguard for division by zero).
    """
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    epsilon = 1e-6
    safe_actual = np.where(actual == 0, epsilon, actual)
    mape = np.mean(np.abs((actual - forecast) / safe_actual)) * 100
    return mae, rmse, mape

def plot_forecast(train, test, forecast, conf_int, plot_path="plots/capital_calls_forecast.png"):
    """
    Plot the training data, test data, forecast and shade the confidence interval.
    Uses the test set's dates for the x-axis.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test (Actual)")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='gray', alpha=0.3, label="Confidence Interval")
    plt.title("Capital Calls Forecast using SARIMAX")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_residuals(residuals, plot_path="plots/residuals.png"):
    """
    Plot residuals along with their ACF, PACF, Q–Q plot, and histogram.
    Also performs the Ljung–Box test and returns its p-value.
    """
    plt.figure(figsize=(12, 12))
    
    plt.subplot(221)
    plt.plot(residuals)
    plt.title("Residuals")
    
    plt.subplot(222)
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title("Q–Q Plot")
    
    plt.subplot(223)
    plot_acf(residuals, ax=plt.gca())
    plt.title("ACF of Residuals")
    
    plt.subplot(224)
    plt.hist(residuals, bins=30)
    plt.title("Residuals Histogram")
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    return plot_path, lb_pvalue

def fit_exponential_smoothing(ts, forecast_steps, seasonal_period=7):
    """
    Fit an Exponential Smoothing model (Holt-Winters) as an alternative forecast.
    """
    model = ExponentialSmoothing(ts, seasonal='add', seasonal_periods=seasonal_period)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(forecast_steps)
    return forecast

def main():
    # Ensure plots directory exists.
    os.makedirs("plots", exist_ok=True)
    
    # Load and clean main data.
    df = load_master_data()
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found.")
    
    # Split into training and testing sets.
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_y = train_df["capital_calls"]
    test_y = test_df["capital_calls"]
    
    # Load exogenous variables (if available) and merge with training data.
    exo = load_exogenous_data()
    if exo is not None:
        # Merge exogenous variables on dates.
        train_exog = train_df.merge(exo, left_index=True, right_index=True, how="left")["10Y Treasury Yield"]
        test_exog = test_df.merge(exo, left_index=True, right_index=True, how="left")["10Y Treasury Yield"]
    else:
        train_exog = test_exog = None
    
    # Check skewness and apply log transformation if skewed.
    skewness = train_y.skew()
    if skewness > 1:
        log_info(f"High skewness ({skewness}) detected; applying log transformation.")
        train_y_transformed = np.log(train_y)
        test_y_transformed = np.log(test_y)
        if train_exog is not None:
            train_exog = np.log(train_exog.replace(0, 1e-6))
            test_exog = np.log(test_exog.replace(0, 1e-6))
        transform_applied = True
    else:
        train_y_transformed = train_y
        test_y_transformed = test_y
        transform_applied = False
    
    # Select best SARIMAX orders (including seasonal) using grid search.
    order, seasonal_order = select_sarimax_order(train_y_transformed, exog=train_exog)
    
    # Fit the SARIMAX model.
    model = SARIMAX(train_y_transformed,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=train_exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Forecast with SARIMAX (using exog from test if available).
    n_forecast = len(test_df)
    forecast_obj = results.get_forecast(steps=n_forecast, exog=test_exog)
    forecast_transformed = forecast_obj.predicted_mean
    conf_int_transformed = forecast_obj.conf_int()
    # Re-index forecasts to test dates.
    forecast_transformed.index = test_df.index
    conf_int_transformed.index = test_df.index
    
    # Invert log transform if applied.
    if transform_applied:
        forecast_sarimax = np.exp(forecast_transformed)
        conf_int = np.exp(conf_int_transformed)
    else:
        forecast_sarimax = forecast_transformed
        conf_int = conf_int_transformed
    
    # Fit an alternative model using Exponential Smoothing.
    forecast_es = fit_exponential_smoothing(train_y, n_forecast)
    forecast_es.index = test_df.index  # Align forecast dates.
    
    # Ensemble forecast: simple average of SARIMAX and Exponential Smoothing forecasts.
    forecast_ensemble = (forecast_sarimax + forecast_es) / 2
    
    # Calculate error metrics on the original scale.
    mae, rmse, mape = calculate_error_metrics(test_y.values, forecast_ensemble.values)
    log_info(f"Ensemble forecast error metrics -- MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    
    # Plot forecast (ensemble forecast with confidence intervals from SARIMAX).
    forecast_plot_path = plot_forecast(train_y, test_y, forecast_ensemble, conf_int)
    
    # Plot residual diagnostics.
    residuals_plot_path, lb_pvalue = plot_residuals(results.resid)
    
    # Build comprehensive JSON output.
    output = {
        "summary": results.summary().as_text(),
        "error_metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        },
        "forecast": forecast_ensemble.to_dict(),
        "confidence_intervals": conf_int.to_dict(),
        "forecast_plot": forecast_plot_path,
        "residuals_plot": residuals_plot_path,
        "ljung_box_pvalue": lb_pvalue,
        "model_order": {
            "nonseasonal_order": order,
            "seasonal_order": seasonal_order
        },
        "transformation": "log" if transform_applied else "none",
        "ensemble_method": "simple average of SARIMAX and Exponential Smoothing"
    }
    return output

def run():
    return main()

if __name__ == "__main__":
    output = run()
    # Ensure that any Timestamp keys are converted to strings for JSON serialization.
    if isinstance(output.get("forecast"), dict):
        output["forecast"] = {str(k): v for k, v in output["forecast"].items()}
    if isinstance(output.get("confidence_intervals"), dict):
        for bound in output["confidence_intervals"]:
            output["confidence_intervals"][bound] = {str(k): v for k, v in output["confidence_intervals"][bound].items()}
    print(json.dumps(output, indent=4))
