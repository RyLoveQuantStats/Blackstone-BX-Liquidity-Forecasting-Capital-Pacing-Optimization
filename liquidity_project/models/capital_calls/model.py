import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, store_dataframe, DB_PATH

# Set up logging and output directories
setup_logging()
OUTPUT_DIR = "output"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MASTER_CSV = os.path.join(CSV_DIR, "synthetic_master_data.csv")
FORECAST_CSV = os.path.join(CSV_DIR, "capital_calls_forecast_rolling.csv")
CORR_CSV = os.path.join(CSV_DIR, "synthetic_correlation.csv")
RISK_PLOT_PATH = os.path.join(PLOT_DIR, "risk_metrics_plot.png")
FORECAST_PLOT_PATH = os.path.join(PLOT_DIR, "forecast_plot.png")

# ---------------------------
# Step 1: Create Synthetic Data
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
    # capital_call_proxy = max(0, (fin_investing_cash_flow + fin_financing_cash_flow)
    #                              - (fin_operating_cash_flow + cash_and_cash_equivalents))
    data['capital_call_proxy'] = (
        data['fin_investing_cash_flow'] + data['fin_financing_cash_flow'] -
        data['fin_operating_cash_flow'] - data['cash_and_cash_equivalents']
    ).apply(lambda x: x if x > 0 else 0)
    
    return data

def save_synthetic_data(data):
    """
    Saves the synthetic master data to CSV and stores it in a new SQL table.
    """
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
# Step 2: Compute Correlation & Save Heatmap
# ---------------------------
def compute_and_save_correlation(data):
    """
    Computes the correlation matrix for the synthetic data,
    saves it as CSV, and generates a heatmap plot saved to disk.
    """
    corr_matrix = data.corr()
    corr_matrix.to_csv(CORR_CSV)
    log_info(f"Correlation matrix saved as CSV in '{CORR_CSV}'.")
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Synthetic Master Data")
    heatmap_path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    log_info(f"Correlation heatmap saved to '{heatmap_path}'.")
    return corr_matrix

# ---------------------------
# Step 3: Compute Capital Call Proxy
# ---------------------------
def compute_capital_call_proxy(df):
    """
    Computes a proxy for capital calls.
    For example:
      capital_call_proxy = max(0, (fin_investing_cash_flow + fin_financing_cash_flow)
                                 - (fin_operating_cash_flow + cash_and_cash_equivalents))
    """
    df = df.copy()
    required_cols = ["fin_investing_cash_flow", "fin_financing_cash_flow", 
                     "fin_operating_cash_flow", "cash_and_cash_equivalents"]
    for col in required_cols:
        if col not in df.columns:
            log_info(f"Column '{col}' missing; defaulting to 0.")
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    df["capital_call_proxy"] = (
        (df["fin_investing_cash_flow"] + df["fin_financing_cash_flow"]) -
        (df["fin_operating_cash_flow"] + df["cash_and_cash_equivalents"])
    ).apply(lambda x: x if x > 0 else 0)
    log_info("Capital call proxy computed.")
    return df

# ---------------------------
# Step 4: Forecasting Using Rolling Exponential Smoothing
# ---------------------------
def forecast_rolling_expsmoothing(data, forecast_horizon=30, window=90):
    """
    Uses a rolling forecasting approach:
      - For each forecast day, fit a SimpleExpSmoothing model on the last 'window' days of the capital_call_proxy.
      - Forecast the next day.
      - Append the forecasted value to the series (for recursive forecasting).
    Returns a DataFrame with forecasted values.
    """
    series = data["capital_call_proxy"].copy()
    forecast_values = []
    for i in range(forecast_horizon):
        recent_data = series[-window:] if len(series) >= window else series
        try:
            model = SimpleExpSmoothing(recent_data).fit()
            forecast = model.forecast(1)
            next_value = forecast.iloc[0]
        except Exception as e:
            log_error(f"Error forecasting at step {i}: {e}")
            next_value = 0
        next_value = max(next_value, 0)
        forecast_values.append(next_value)
        # Use pd.concat instead of series.append (since append is deprecated)
        next_series = pd.Series([next_value], index=[series.index[-1] + timedelta(days=1)])
        series = pd.concat([series, next_series])
    
    last_date = data.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D")
    forecast_df = pd.DataFrame({"forecast_capital_calls": forecast_values}, index=forecast_index)
    return forecast_df

# ---------------------------
# Step 5: Compute Risk Metrics
# ---------------------------
def compute_risk_metrics(forecast_df, residual_std):
    """
    Computes simple risk metrics for the forecast.
    VaR_5 = forecast - 1.645 * residual_std
    CVaR_5 = forecast - 1.96 * residual_std
    """
    forecast_df = forecast_df.copy()
    forecast_df["VaR_5"] = forecast_df["forecast_capital_calls"] - 1.645 * residual_std
    forecast_df["CVaR_5"] = forecast_df["forecast_capital_calls"] - 1.96 * residual_std
    forecast_df["VaR_5"] = forecast_df["VaR_5"].apply(lambda x: x if x > 0 else 0)
    forecast_df["CVaR_5"] = forecast_df["CVaR_5"].apply(lambda x: x if x > 0 else 0)
    return forecast_df

# ---------------------------
# Step 6: Store Forecast & Statistics in SQL
# ---------------------------
def store_forecast_and_stats(forecast_df, data, corr_matrix, residual_std):
    """
    Stores forecast results, synthetic master data, and risk statistics in new SQL tables.
    """
    try:
        conn = get_connection()
        forecast_df.to_sql("capital_calls_forecast_rolling", conn, if_exists="replace", index=True, index_label="Date")
        data.to_sql("synthetic_master_data", conn, if_exists="replace", index=True, index_label="Date")
        corr_matrix.to_sql("synthetic_correlation", conn, if_exists="replace", index=True, index_label="Variable")
        risk_stats = pd.DataFrame({
            "residual_std": [residual_std],
            "forecast_mean": [forecast_df["forecast_capital_calls"].mean()],
            "forecast_std": [forecast_df["forecast_capital_calls"].std()]
        })
        risk_stats.to_sql("forecast_risk_stats", conn, if_exists="replace", index=False)
        conn.close()
        log_info("Forecast, synthetic master data, and risk stats stored successfully in the database.")
    except Exception as e:
        log_error(f"Error storing forecast and stats in database: {e}")
        raise

# ---------------------------
# Step 7: Plot Forecast and Risk Metrics
# ---------------------------
def plot_forecast_and_risk(data, forecast_df):
    """
    Plots the historical capital call proxy and forecast, and overlays risk metrics.
    Saves the plot to disk.
    """
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data["capital_call_proxy"], label="Historical Capital Call Proxy")
    plt.plot(forecast_df.index, forecast_df["forecast_capital_calls"], label="Forecast", color="red")
    plt.plot(forecast_df.index, forecast_df["VaR_5"], label="VaR 5%", color="orange", linestyle="--")
    plt.plot(forecast_df.index, forecast_df["CVaR_5"], label="CVaR 5%", color="purple", linestyle="--")
    plt.title("Forecast of Capital Calls with Risk Metrics")
    plt.xlabel("Date")
    plt.ylabel("Capital Calls")
    plt.legend()
    plot_path = os.path.join(PLOT_DIR, "forecast_risk_plot.png")
    plt.savefig(plot_path)
    plt.show()
    log_info(f"Forecast and risk metrics plot saved to '{plot_path}'.")

# ---------------------------
# Main Execution
# ---------------------------
def main():
    # Step 1: Create synthetic master data and store it.
    synthetic_data = create_synthetic_master_data()
    synthetic_data = save_synthetic_data(synthetic_data)
    
    # Step 2: Compute and save correlation statistics.
    corr_matrix = synthetic_data.corr()
    corr_matrix.to_csv(CORR_CSV)
    log_info(f"Correlation matrix saved as CSV in '{CORR_CSV}'.")
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Synthetic Master Data")
    heatmap_path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    log_info(f"Correlation heatmap saved to '{heatmap_path}'.")
    
    # Step 3: Compute capital call proxy.
    synthetic_data = compute_capital_call_proxy(synthetic_data)
    
    # Step 4: Forecast future capital calls using a rolling exponential smoothing model.
    forecast_df = forecast_rolling_expsmoothing(synthetic_data, forecast_horizon=30, window=90)
    
    # For risk metrics, approximate forecast error using the standard deviation of residuals
    recent_data = synthetic_data["capital_call_proxy"][-90:]
    try:
        model = SimpleExpSmoothing(recent_data).fit()
        residuals = recent_data - model.fittedvalues
        residual_std = np.std(residuals)
    except Exception as e:
        log_error(f"Error computing residuals for risk metrics: {e}")
        residual_std = 0
    
    # Step 5: Compute risk metrics on the forecast.
    forecast_df = compute_risk_metrics(forecast_df, residual_std)
    
    # Step 6: Save forecast to CSV.
    forecast_df.to_csv(FORECAST_CSV)
    log_info(f"Forecast saved as CSV in '{FORECAST_CSV}'.")
    
    # Step 7: Store forecast, synthetic data, and risk statistics in SQL.
    store_forecast_and_stats(forecast_df, synthetic_data, corr_matrix, residual_std)
    
    # Step 8: Plot forecast and risk metrics.
    plot_forecast_and_risk(synthetic_data, forecast_df)
    
    return forecast_df

if __name__ == "__main__":
    forecast_results = main()
    print(forecast_results.tail(10))
