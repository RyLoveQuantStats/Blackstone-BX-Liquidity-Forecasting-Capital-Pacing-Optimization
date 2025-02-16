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


# ---------------------------
# Model2: Monte Carlo Simulation
# ---------------------------

#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import json
from datetime import timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import DB and logging utilities from your utils folder
from capital_calls_api.utils.db_utils import get_connection, DB_PATH, store_dataframe
from capital_calls_api.utils.logging_utils import setup_logging, log_info, log_error

# Set up logging (file + console)
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Global constants and output paths
TABLE_NAME = "synthetic_master_data"
OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ===============================
# Forecasting Functions
# ===============================

def load_exogenous_data(db_path=DB_PATH):
    """
    Load exogenous data (10Y Treasury Yield) from the macroeconomic_data table.
    Returns None if not available.
    """
    try:
        conn = get_connection()
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
    Loads the master data, drops the 'Date' column,
    and uses the 'index' column as the actual Date index.
    """
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()

        # 1. Drop 'Date' if it exists
        if "Date" in df.columns:
            df.drop(columns=["Date"], inplace=True)

        # 2. Rename 'index' -> 'Date'
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)

        # 3. Convert 'Date' to datetime and set as index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

        # 4. Force daily frequency if not inferred
        if pd.infer_freq(df.index) is None:
            df = df.asfreq("D")
            log_info("No frequency inferred; defaulting to daily.")

        # 5. Check for capital_call_proxy
        if "capital_call_proxy" not in df.columns:
            return None

        return df
    except Exception as e:
        log_error(f"Error loading master data: {e}")
        return None

def is_stationary(series, alpha=0.05):
    """Checks if a series is stationary using the ADF test."""
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        log_info(f"ADF p-value = {p_value}")
        return p_value < alpha
    except Exception as e:
        log_error(f"ADF test failed: {e}")
        return False

def calculate_errors(actual, forecast):
    """Calculates MAE, RMSE, and MAPE."""
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast)**2))
    mape = np.mean(np.abs((actual - forecast) / np.where(actual == 0, 1e-6, actual))) * 100
    return mae, rmse, mape

def plot_forecast_forecasting(train, test, forecast, conf_int,
                              save_path=os.path.join(PLOTS_DIR, "capital_calls_forecast.png")):
    """Plots the train, test, and forecast series with confidence intervals."""
    plt.figure(figsize=(10,5))
    plt.plot(train, label="Train", alpha=0.8)
    plt.plot(test, label="Test", alpha=0.8)
    plt.plot(forecast, label="Forecast (Ensemble)", linestyle="--", color="red")
    if conf_int is not None:
        plt.fill_between(forecast.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                         color="gray", alpha=0.3, label="Confidence Interval")
    plt.title("Capital Calls Forecast")
    plt.grid(True)  # Enhancement: add grid
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_residuals_forecasting(resid,
                               save_path=os.path.join(PLOTS_DIR, "residuals.png")):
    """
    Plots residuals, ACF, PACF, Q-Q plot; returns plot path & Ljung-Box p-value.
    """
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.plot(resid, alpha=0.8)
    plt.title("Residuals")
    plt.grid(True)

    plt.subplot(222)
    qqplot(resid, line="s", ax=plt.gca())
    plt.title("Q-Q Plot")
    plt.grid(True)

    plt.subplot(223)
    plot_acf(resid.dropna(), ax=plt.gca())
    plt.title("ACF")
    plt.grid(True)

    plt.subplot(224)
    plot_pacf(resid.dropna(), ax=plt.gca(), method="ywm")
    plt.title("PACF")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
    pvalue = lb["lb_pvalue"].values[0]
    return save_path, pvalue

def run_forecasting():
    """
    Main forecasting function.
    1. Load master data
    2. Check stationarity & difference if needed
    3. Split into train/test
    4. Fit SARIMAX & Exponential Smoothing
    5. Average forecasts -> ensemble
    6. Compute error metrics & produce plots
    7. Return an output dictionary
    """
    df = load_master_data()
    if df is None or "capital_call_proxy" not in df.columns:
        return {"error": "capital_call_proxy column not found or data load failed."}

    df = df[df["capital_call_proxy"].notnull()]
    series = df["capital_call_proxy"]
    if series.nunique() <= 1:
        log_error("capital_calls is constant. Returning trivial forecast.")
        return {"error": "constant data", "trivial_forecast": float(series.iloc[-1]) if len(series) > 0 else 0.0}

    # Stationarity check
    if not is_stationary(series):
        log_info("Series not stationary; differencing once.")
        series = series.diff().dropna()

    # Train/Test split
    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    # If exogenous data is present
    if "10Y Treasury Yield" in df.columns:
        exog = df["10Y Treasury Yield"]
        train_exog = exog.iloc[:train_size]
        test_exog = exog.iloc[train_size:]
    else:
        train_exog = test_exog = None

    # Fit SARIMAX
    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 7)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    exog=train_exog, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Forecast using SARIMAX
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

    # Ensemble
    forecast_ensemble = (forecast_sarimax + forecast_es) / 2

    # Error metrics
    mae, rmse, mape = calculate_errors(test.values, forecast_ensemble.values)
    log_info(f"Forecast Errors: MAE={mae}, RMSE={rmse}, MAPE={mape}")

    # Plot
    forecast_plot = plot_forecast_forecasting(train, test, forecast_ensemble, conf_int)
    residuals_plot, lb_pvalue = plot_residuals_forecasting(results.resid.dropna())

    output = {
        "model_summary": results.summary().as_text(),
        "error_metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape},
        "forecast": forecast_ensemble.to_dict(),       # keys = Timestamps
        "confidence_intervals": conf_int.to_dict(),    # keys = Timestamps
        "forecast_plot": forecast_plot,
        "residuals_plot": residuals_plot,
        "ljung_box_pvalue": lb_pvalue,
        "model_order": {"order": order, "seasonal_order": seasonal_order},
        "transformation": "diff once if non-stationary",
        # Enhancement: metadata about date range
        "data_range": {
            "train_start": str(train.index[0]),
            "train_end": str(train.index[-1]),
            "test_start": str(test.index[0]),
            "test_end": str(test.index[-1])
        }
    }
    return output

# ===============================
# Simulation Functions
# ===============================

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

def simulate_shocks(method, mean_change, std_change, historical_changes,
                    n_simulations, horizon, stress_mean=1.0, stress_std=1.0,
                    block_size=5, rolling_window=90):
    """
    Creates random shocks using various methods (normal, t, bootstrap, etc.)
    """
    if method == "normal":
        shocks = np.random.normal(mean_change * stress_mean, std_change * stress_std,
                                  (n_simulations, horizon))
    elif method == "t":
        df_t = 5
        scale_factor = std_change * stress_std / np.sqrt(df_t / (df_t - 2))
        shocks = np.random.standard_t(df_t, (n_simulations, horizon)) * scale_factor + mean_change * stress_mean
    elif method == "bootstrap":
        shocks = np.random.choice(historical_changes, size=(n_simulations, horizon), replace=True)
    elif method == "kde":
        kde = stats.gaussian_kde(historical_changes)
        shocks = kde.resample(n_simulations * horizon).T.reshape(n_simulations, horizon)
        shocks = shocks * stress_std + mean_change * stress_mean
    elif method == "block":
        n = len(historical_changes)
        blocks_per_sim = int(np.ceil(horizon / block_size))
        shocks = np.empty((n_simulations, horizon))
        for i in range(n_simulations):
            indices = np.random.randint(0, n - block_size + 1, blocks_per_sim)
            sim_shocks = np.concatenate([historical_changes[idx:idx+block_size] for idx in indices])
            shocks[i, :] = sim_shocks[:horizon]
        shocks = shocks * stress_std + mean_change * stress_mean
    elif method == "rolling":
        recent = historical_changes[-rolling_window:]
        r_mean = np.mean(recent)
        r_std = np.std(recent)
        shocks = np.random.normal(r_mean * stress_mean, r_std * stress_std, (n_simulations, horizon))
    elif method == "garch":
        if not ARCH_AVAILABLE:
            raise ImportError("arch package not available. Use a different method.")
        am = arch_model(historical_changes, vol='Garch', p=1, o=0, q=1, dist='normal')
        res = am.fit(disp='off')
        shocks = np.empty((n_simulations, horizon))
        for i in range(n_simulations):
            sim_data = res.simulate(res.params, horizon)
            shocks[i, :] = sim_data['data']
            shocks[i, :] = shocks[i, :] * stress_std + mean_change * stress_mean
    else:
        raise ValueError("Unsupported simulation method.")
    return shocks

def run_monte_carlo_simulation(df, n_simulations=100000, horizon=30, method="normal",
                               stress_mean=1.0, stress_std=1.0, block_size=5,
                               rolling_window=90, parallel=False):
    """
    Runs Monte Carlo simulations on capital_call_proxy using the chosen method.
    """
    df["calls_change"] = df["capital_call_proxy"].diff().fillna(0)
    historical_changes = df["calls_change"].values
    mean_change = np.mean(historical_changes)
    std_change = np.std(historical_changes)
    log_info(f"Mean daily change: {mean_change:.2f}, Std: {std_change:.2f}")
    last_value = df["capital_call_proxy"].iloc[-1]

    # Possibly parallelize
    if parallel and PARALLEL_AVAILABLE:
        n_jobs = -1
        chunk_size = n_simulations // 10
        simulation_results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_shocks)(
                method, mean_change, std_change, historical_changes,
                chunk_size, horizon, stress_mean, stress_std, block_size, rolling_window
            )
            for _ in range(10)
        )
        shocks_sum = np.concatenate(simulation_results).sum(axis=1)
    else:
        shocks = simulate_shocks(
            method, mean_change, std_change, historical_changes,
            n_simulations, horizon, stress_mean, stress_std, block_size, rolling_window
        )
        shocks_sum = shocks.sum(axis=1)

    simulated_outcomes = last_value + shocks_sum
    simulated_outcomes = np.clip(simulated_outcomes, 0, None)
    return simulated_outcomes, mean_change, std_change, last_value

def calculate_risk_metrics_sim(outcomes, percentile=5):
    """
    Calculates VaR, CVaR, skewness, and kurtosis for simulation outcomes.
    """
    VaR = np.percentile(outcomes, percentile)
    CVaR = outcomes[outcomes <= VaR].mean() if np.any(outcomes <= VaR) else None
    skewness = float(stats.skew(outcomes))
    kurtosis = float(stats.kurtosis(outcomes))
    return VaR, CVaR, skewness, kurtosis

def plot_simulation_histogram(outcomes,
                              plot_path=os.path.join(PLOTS_DIR, "monte_carlo_histogram.png")):
    plt.figure(figsize=(10, 5))
    plt.hist(outcomes, bins=50, color='skyblue', edgecolor='black', alpha=0.75)
    plt.title("Distribution of Simulated Capital Calls Outcomes")
    plt.xlabel("Simulated Capital Calls")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_sample_paths(outcomes, n_paths=20, horizon=30,
                      plot_path=os.path.join(PLOTS_DIR, "monte_carlo_sample_paths.png")):
    """
    Plots sample simulation paths.
    """
    plt.figure(figsize=(10, 5))
    time_axis = np.arange(1, horizon + 1)
    for i in range(min(n_paths, len(outcomes))):
        path = np.linspace(0, outcomes[i], horizon)
        plt.plot(time_axis, path, lw=1, alpha=0.7)
    plt.title(f"Sample Simulation Paths (n={min(n_paths, len(outcomes))})")
    plt.xlabel("Days Ahead")
    plt.ylabel("Simulated Capital Calls")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def run_simulation(args):
    """
    Main simulation function.
    1. Load master data
    2. Monte Carlo simulation with chosen method
    3. Optional stress/sensitivity analysis
    4. Plots & risk metrics
    5. Return an output dictionary
    """
    df = load_master_data()
    if "capital_call_proxy" not in df.columns:
        raise ValueError("❌ 'capital_call_proxy' column not found.")

    outcomes, mean_change, std_change, last_value = run_monte_carlo_simulation(
        df,
        n_simulations=args.n_simulations,
        horizon=args.horizon,
        method=args.method,
        stress_mean=args.stress_mean,
        stress_std=args.stress_std,
        block_size=args.block_size,
        rolling_window=args.rolling_window,
        parallel=args.parallel
    )

    # Sensitivity analysis
    stress_mean_list = [float(x) for x in args.stress_mean_list.split(",")]
    stress_std_list = [float(x) for x in args.stress_std_list.split(",")]
    sensitivity_results = {}
    if len(stress_mean_list) > 1 or len(stress_std_list) > 1:
        for sm in stress_mean_list:
            for ss in stress_std_list:
                sim_outcomes, _, _, _ = run_monte_carlo_simulation(
                    df,
                    n_simulations=args.n_simulations // 10,
                    horizon=args.horizon,
                    method=args.method,
                    stress_mean=sm,
                    stress_std=ss,
                    block_size=args.block_size,
                    rolling_window=args.rolling_window,
                    parallel=False
                )
                p5 = float(np.percentile(sim_outcomes, 5))
                p50 = float(np.percentile(sim_outcomes, 50))
                p95 = float(np.percentile(sim_outcomes, 95))
                VaR_, CVaR_, skew_, kurt_ = calculate_risk_metrics_sim(sim_outcomes)
                sensitivity_results[f"stress_mean={sm}_stress_std={ss}"] = {
                    "5th_percentile": p5,
                    "median": p50,
                    "95th_percentile": p95,
                    "VaR": VaR_,
                    "CVaR": CVaR_,
                    "skewness": skew_,
                    "kurtosis": kurt_
                }

    # Risk metrics
    p5 = float(np.percentile(outcomes, 5))
    p50 = float(np.percentile(outcomes, 50))
    p95 = float(np.percentile(outcomes, 95))
    VaR, CVaR, sim_skew, sim_kurt = calculate_risk_metrics_sim(outcomes)
    log_info(f"Simulation percentiles -- 5th: {p5:.2f}, Median: {p50:.2f}, 95th: {p95:.2f}")
    log_info(f"VaR: {VaR:.2f}, CVaR: {CVaR}, Skewness: {sim_skew:.2f}, Kurtosis: {sim_kurt:.2f}")

    hist_plot_path = plot_simulation_histogram(outcomes)
    sample_paths_plot_path = plot_sample_paths(outcomes, horizon=args.horizon)

    output = {
        "mean_daily_change": mean_change,
        "std_daily_change": std_change,
        "last_capital_calls": last_value,
        "5th_percentile": p5,
        "median": p50,
        "95th_percentile": p95,
        "VaR": VaR,
        "CVaR": CVaR,
        "skewness": sim_skew,
        "kurtosis": sim_kurt,
        "n_simulations": args.n_simulations,
        "horizon_days": args.horizon,
        "simulation_method": args.method,
        "stress_mean_factor": args.stress_mean,
        "stress_std_factor": args.stress_std,
        "histogram_plot": hist_plot_path,
        "sample_paths_plot": sample_paths_plot_path,
        "sensitivity_analysis": sensitivity_results,
        # Enhancement: data range
        "data_range": {
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "rows": len(df)
        }
    }
    return output
