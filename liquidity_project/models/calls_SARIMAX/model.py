import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import logging
import json

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.optimize import minimize

from capital_calls_api.utils.db_utils import get_connection, store_dataframe, DB_PATH
from capital_calls_api.utils.logging_utils import setup_logging, log_info, log_error

# Set up logging (both file and console)
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Output paths
OUTPUT_DIR = "output"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# File paths for outputs
FORECAST_CSV = os.path.join(CSV_DIR, "capital_calls_forecast_sarimax.csv")
CORR_CSV = os.path.join(CSV_DIR, "synthetic_correlation.csv")
HEATMAP_PNG = os.path.join(PLOT_DIR, "correlation_heatmap.png")
FORECAST_PLOT_PNG = os.path.join(PLOT_DIR, "forecast_plot.png")


# ---------------------------
# Synthetic Data Generation
# ---------------------------
def create_synthetic_master_data():
    """
    Creates a synthetic dataset representing daily financial data from 2020 to 2024.
    Mimics key financial metrics for capital call forecasting.
    """
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    np.random.seed(42)
    data = pd.DataFrame({
        'fin_investing_cash_flow': np.random.normal(loc=100, scale=50, size=n),
        'fin_financing_cash_flow': np.random.normal(loc=50, scale=25, size=n),
        'fin_operating_cash_flow': np.random.normal(loc=200, scale=75, size=n),
        'cash_and_cash_equivalents': np.random.normal(loc=300, scale=100, size=n),
        'macro_10Y Treasury Yield': np.random.normal(loc=2, scale=0.5, size=n)
    }, index=dates)
    
    # Compute a capital call proxy:
    data['capital_call_proxy'] = (
        data['fin_investing_cash_flow'] + data['fin_financing_cash_flow'] -
        data['fin_operating_cash_flow'] - data['cash_and_cash_equivalents']
    ).apply(lambda x: x if x > 0 else 0)
    
    return data

def save_synthetic_data(data):
    """
    Saves the synthetic master data to CSV and stores it in a new SQL table.
    """
    MASTER_CSV = os.path.join(CSV_DIR, "synthetic_master_data.csv")
    data.to_csv(MASTER_CSV)
    log_info(f"Synthetic master data saved as CSV in '{MASTER_CSV}'.")
    try:
        store_dataframe(data, "synthetic_master_data", if_exists="replace")
        log_info("Synthetic master data stored in SQL table 'synthetic_master_data'.")
    except Exception as e:
        log_error(f"Error storing synthetic master data: {e}")
        raise
    return data


# ---------------------------
# Data Loading Function (Revised)
# ---------------------------
def load_synthetic_data():
    """
    Loads the synthetic_master_data table from the database.
    If the 'Date' column exists, it will be converted to datetime and set as the index.
    Otherwise, it assumes that the index already holds the date information.
    """
    try:
        conn = get_connection()
        # Load without specifying index_col or parse_dates
        df = pd.read_sql("SELECT * FROM synthetic_master_data", conn)
        conn.close()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        else:
            # Assume the index was stored already as the date
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                log_error("Failed to convert index to datetime.", e)
        log_info(f"Synthetic master data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading synthetic master data: {e}")
        raise


# ---------------------------
# Forecasting Function (SARIMAX)
# ---------------------------
def forecast_capital_call_proxy():
    """
    Uses a SARIMAX model to forecast future values of the 'capital_call_proxy' column
    from the synthetic_master_data. Returns the 12-month forecast, historical series,
    forecast series, and confidence intervals.
    """
    try:
        df = load_synthetic_data()
        # Ensure we have the capital_call_proxy column
        if "capital_call_proxy" not in df.columns:
            log_error("Column 'capital_call_proxy' not found in synthetic_master_data.")
            return 0, None, None, None
        
        series = df["capital_call_proxy"].copy()
        # If frequency is not set, assume monthly start ('MS')
        if series.index.inferred_freq is None:
            series = series.asfreq('MS')
        
        # Fit SARIMAX model - example order; adjust if necessary
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast_steps = 12
        forecast_obj = results.get_forecast(steps=forecast_steps)
        forecast_series = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        forecast_sum = float(forecast_series.sum())
        log_info(f"Forecasted capital call proxy (12-month sum): {forecast_sum}")
        return forecast_sum, series, forecast_series, conf_int
    except Exception as e:
        log_error(f"Error forecasting capital call proxy: {e}")
        return 0, None, None, None


# ---------------------------
# Risk Metrics Function
# ---------------------------
def compute_risk_metrics(forecast_series, residual_std):
    """
    Computes simple risk metrics for the forecast:
      VaR_5 = forecast - 1.645 * residual_std
      CVaR_5 = forecast - 1.96 * residual_std
    """
    df_forecast = forecast_series.to_frame(name="forecast_capital_calls")
    df_forecast["VaR_5"] = df_forecast["forecast_capital_calls"] - 1.645 * residual_std
    df_forecast["CVaR_5"] = df_forecast["forecast_capital_calls"] - 1.96 * residual_std
    df_forecast["VaR_5"] = df_forecast["VaR_5"].apply(lambda x: x if x > 0 else 0)
    df_forecast["CVaR_5"] = df_forecast["CVaR_5"].apply(lambda x: x if x > 0 else 0)
    return df_forecast


# ---------------------------
# Plotting Function
# ---------------------------
def plot_forecast(history, forecast_series, conf_int):
    """
    Plots historical capital_call_proxy and forecast with confidence intervals.
    Saves the plot to disk.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history, label="Historical Capital Call Proxy", color="blue")
    plt.plot(forecast_series.index, forecast_series, label="Forecast", color="red")
    if conf_int is not None:
        plt.fill_between(forecast_series.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                         color='gray', alpha=0.3, label="Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Capital Call Proxy")
    plt.title("Forecast of Capital Call Proxy (SARIMAX)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PNG)
    plt.show()
    log_info(f"Forecast plot saved to '{FORECAST_PLOT_PNG}'.")


# ---------------------------
# Store Analysis Results in SQL
# ---------------------------
def store_analysis_results(forecast_df, synthetic_df, corr_matrix, risk_stats):
    """
    Stores forecast results, synthetic master data, correlation matrix, and risk statistics
    into new SQL tables.
    """
    try:
        conn = get_connection()
        forecast_df.to_sql("capital_calls_forecast", conn, if_exists="replace", index=True, index_label="Date")
        synthetic_df.to_sql("synthetic_master_data", conn, if_exists="replace", index=True, index_label="Date")
        corr_matrix.to_sql("synthetic_correlation", conn, if_exists="replace", index=True, index_label="Variable")
        risk_stats.to_sql("forecast_risk_stats", conn, if_exists="replace", index=False)
        conn.close()
        log_info("Forecast, synthetic data, correlation, and risk stats stored in the database.")
    except Exception as e:
        log_error(f"Error storing analysis results: {e}")
        raise


# ---------------------------
# Main Execution Function
# ---------------------------
def main():
    os.makedirs("output", exist_ok=True)
    
    # Create synthetic master data and store it in the DB
    synthetic_data = create_synthetic_master_data()
    synthetic_data = save_synthetic_data(synthetic_data)
    
    # Load synthetic master data (now updated in the DB)
    synthetic_df = load_synthetic_data()
    
    # Compute correlation matrix using only numeric columns and save heatmap
    numeric_df = synthetic_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    corr_matrix.to_csv(CORR_CSV)
    log_info(f"Correlation matrix saved to '{CORR_CSV}'.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Synthetic Master Data")
    plt.savefig(HEATMAP_PNG)
    plt.close()
    log_info(f"Correlation heatmap saved to '{HEATMAP_PNG}'.")
    
    # Forecast capital call proxy using SARIMAX
    forecast_sum, hist_series, forecast_series, conf_int = forecast_capital_call_proxy()
    
    # Plot forecast if available
    if hist_series is not None and forecast_series is not None:
        plot_forecast(hist_series, forecast_series, conf_int)
    
    # Compute risk metrics using residual standard deviation from a SimpleExpSmoothing model
    try:
        recent_data = hist_series[-90:]  # last 90 data points
        model_fit = SimpleExpSmoothing(recent_data).fit()
        residuals = recent_data - model_fit.fittedvalues
        residual_std = np.std(residuals)
    except Exception as e:
        log_error(f"Error computing residuals: {e}")
        residual_std = 0

    forecast_risk_df = compute_risk_metrics(forecast_series, residual_std)
    
    # Create a simple risk statistics DataFrame
    risk_stats = pd.DataFrame({
        "residual_std": [residual_std],
        "forecast_mean": [forecast_series.mean()],
        "forecast_std": [forecast_series.std()]
    })
    
    # Store forecast and analysis results in SQL
    store_analysis_results(forecast_series.to_frame(name="forecast_capital_calls"),
                             synthetic_df, corr_matrix, risk_stats)
    
    # Export forecast to CSV
    forecast_series.to_frame(name="forecast_capital_calls").to_csv(FORECAST_CSV)
    log_info(f"Forecast saved as CSV in '{FORECAST_CSV}'.")
    
    return forecast_series

if __name__ == "__main__":
    forecast_results = main()
