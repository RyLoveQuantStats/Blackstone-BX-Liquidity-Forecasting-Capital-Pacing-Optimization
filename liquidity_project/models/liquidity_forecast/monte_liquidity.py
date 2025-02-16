"""
Monte Carlo Simulation for Liquidity Forecasting
---------------------------------------------------------
This script loads the merged KKR dataset from a centralized SQLite database,
and runs advanced Monte Carlo simulations on the 'capital_calls' time series.
It offers several simulation methods (normal, t, bootstrap, garch, kde, block, rolling)
and supports stress testing, parallel processing, and sensitivity analysis.
It computes additional risk metrics (VaR, CVaR, skewness, kurtosis) and produces
diagnostic plots. The output is provided as JSON for API/dashboard integration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy import stats

# Try to import arch for GARCH simulation.
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# For parallel processing
try:
    from joblib import Parallel, delayed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# Import centralized DB and logging utilities.
from utils.db_utils import get_connection, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error
import logging

# Set up logging (both file and console).
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

TABLE_NAME = "synthetic_master_data"
log_info(f"Final DB_PATH = {DB_PATH}")

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    """
    Load the master dataset from the SQLite database.
      - Sets 'Date' as the index.
      - Infers frequency or forces daily frequency.
      - Drops rows with missing 'capital_calls' and removes outliers using a MAD method.
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
            log_info("Could not infer frequency from Date index. Forcing daily frequency ('D').")
            df = df.asfreq('D')
        df = df[df["capital_call_proxy"].notnull()]
        # Remove outliers using Median Absolute Deviation (MAD)
        median = df["capital_call_proxy"].median()
        mad = np.median(np.abs(df["capital_call_proxy"] - median))
        threshold = 3.5
        modified_z = 0.6745 * (df["capital_call_proxy"] - median) / (mad + 1e-6)
        df_clean = df[np.abs(modified_z) < threshold]
        log_info(f"Removed {len(df) - len(df_clean)} outlier rows based on MAD.")
        return df_clean
    except Exception as e:
        log_error(f"Error loading data: {e}")
        raise

def simulate_shocks(method, mean_change, std_change, historical_changes,
                    n_simulations, horizon, stress_mean=1.0, stress_std=1.0,
                    block_size=5, rolling_window=90):
    """
    Simulate daily shocks over a given horizon using one of the methods:
      - 'normal': Normal distribution (with stress multipliers).
      - 't': t-distribution with df=5.
      - 'bootstrap': Resample historical changes.
      - 'kde': Sample from an estimated kernel density.
      - 'block': Block bootstrapping (with given block_size).
      - 'rolling': Use rolling window estimates (mean and std from last rolling_window days).
      - 'garch': Use a fitted GARCH(1,1) model (if available).
    Returns an array of shape (n_simulations, horizon).
    """
    if method == "normal":
        shocks = np.random.normal(mean_change * stress_mean, std_change * stress_std, (n_simulations, horizon))
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
        raise ValueError("Unsupported simulation method. Choose from 'normal', 't', 'bootstrap', 'kde', 'block', 'rolling', or 'garch'.")
    return shocks

def simulate_chunk(n_chunk, method, mean_change, std_change, historical_changes,
                   horizon, stress_mean, stress_std, block_size, rolling_window):
    """Simulate one chunk of n_chunk simulations."""
    return run_monte_carlo_simulation_core(method, mean_change, std_change, historical_changes,
                                             n_chunk, horizon, stress_mean, stress_std, block_size, rolling_window)

def run_monte_carlo_simulation_core(method, mean_change, std_change, historical_changes,
                                    n_simulations, horizon, stress_mean, stress_std, block_size, rolling_window):
    shocks = simulate_shocks(method, mean_change, std_change, historical_changes,
                             n_simulations, horizon, stress_mean, stress_std, block_size, rolling_window)
    return shocks.sum(axis=1)

def run_monte_carlo_simulation(df, n_simulations=100000, horizon=30, method="normal",
                               stress_mean=1.0, stress_std=1.0, block_size=5, rolling_window=90, parallel=False):
    """
    Run Monte Carlo simulations on the capital_calls series.
      - Computes the daily change distribution.
      - Simulates n_simulations paths over a given horizon.
      - Clips outcomes to a minimum of zero.
    Returns:
      simulated_outcomes, mean_change, std_change, last_value.
    """
    df["calls_change"] = df["capital_call_proxy"].diff().fillna(0)
    historical_changes = df["calls_change"].values
    mean_change = np.mean(historical_changes)
    std_change = np.std(historical_changes)
    log_info(f"Mean daily change: {mean_change:.2f}, Std: {std_change:.2f}")
    last_value = df["capital_call_proxy"].iloc[-1]
    
    if parallel and PARALLEL_AVAILABLE:
        n_jobs = -1  # use all available cores
        chunk_size = n_simulations // 10
        simulation_results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_chunk)(chunk_size, method, mean_change, std_change,
                                      historical_changes, horizon, stress_mean, stress_std, block_size, rolling_window)
            for _ in range(10)
        )
        shocks_sum = np.concatenate(simulation_results)
    else:
        shocks_sum = simulate_shocks(method, mean_change, std_change, historical_changes,
                                     n_simulations, horizon, stress_mean, stress_std, block_size, rolling_window).sum(axis=1)
    
    simulated_outcomes = last_value + shocks_sum
    simulated_outcomes = np.clip(simulated_outcomes, 0, None)
    return simulated_outcomes, mean_change, std_change, last_value

def calculate_risk_metrics(outcomes, percentile=5):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    Also compute skewness and kurtosis.
    """
    VaR = np.percentile(outcomes, percentile)
    CVaR = outcomes[outcomes <= VaR].mean() if np.any(outcomes <= VaR) else None
    skewness = float(stats.skew(outcomes))
    kurtosis = float(stats.kurtosis(outcomes))
    return VaR, CVaR, skewness, kurtosis

def plot_simulation_histogram(outcomes, plot_path="plots/monte_carlo_histogram.png"):
    """
    Plot a histogram of simulated outcomes.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(outcomes, bins=50, color='skyblue', edgecolor='black', alpha=0.75)
    plt.title("Distribution of Simulated Capital Calls Outcomes")
    plt.xlabel("Simulated Capital Calls")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_sample_paths(outcomes, n_paths=20, horizon=30, plot_path="plots/monte_carlo_sample_paths.png"):
    """
    Plot a few sample simulation paths over the forecast horizon.
    Here we approximate a cumulative path by interpolating between the average outcome and the simulated final outcome.
    """
    plt.figure(figsize=(10, 5))
    time_axis = np.arange(1, horizon + 1)
    for i in range(n_paths):
        path = np.linspace(0, outcomes[i], horizon)
        plt.plot(time_axis, path, lw=1, alpha=0.7)
    plt.title(f"Sample Simulation Paths (n={n_paths})")
    plt.xlabel("Days Ahead")
    plt.ylabel("Simulated Capital Calls")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main(args):
    os.makedirs("plots", exist_ok=True)
    df = load_master_data()
    if "capital_call_proxy" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found.")
    
    # Run Monte Carlo simulation.
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
    
    # Run sensitivity analysis if multiple stress factors are provided.
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
                VaR, CVaR, skewness, kurtosis = calculate_risk_metrics(sim_outcomes)
                sensitivity_results[f"stress_mean={sm}_stress_std={ss}"] = {
                    "5th_percentile": p5,
                    "median": p50,
                    "95th_percentile": p95,
                    "VaR": VaR,
                    "CVaR": CVaR,
                    "skewness": skewness,
                    "kurtosis": kurtosis
                }
    
    # Compute overall simulation percentiles and risk metrics.
    p5 = float(np.percentile(outcomes, 5))
    p50 = float(np.percentile(outcomes, 50))
    p95 = float(np.percentile(outcomes, 95))
    VaR, CVaR, sim_skew, sim_kurt = calculate_risk_metrics(outcomes)
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
        "sensitivity_analysis": sensitivity_results
    }
    return output

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Monte Carlo simulation for liquidity forecasting.")
    parser.add_argument("--n_simulations", type=int, default=100000, help="Number of simulations")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--method", type=str, default="normal",
                        choices=["normal", "t", "bootstrap", "kde", "block", "rolling", "garch"],
                        help="Method to simulate daily shocks")
    parser.add_argument("--stress_mean", type=float, default=1.0, help="Stress multiplier for mean")
    parser.add_argument("--stress_std", type=float, default=1.0, help="Stress multiplier for std deviation")
    parser.add_argument("--block_size", type=int, default=5, help="Block size for block bootstrapping")
    parser.add_argument("--rolling_window", type=int, default=90, help="Window size for rolling simulation")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--stress_mean_list", type=str, default="1.0",
                        help="Comma-separated list of stress mean factors for sensitivity analysis")
    parser.add_argument("--stress_std_list", type=str, default="1.0",
                        help="Comma-separated list of stress std factors for sensitivity analysis")
    return parser.parse_args()

def run():
    args = parse_args()
    result = main(args)
    return result

if __name__ == "__main__":
    output = run()
    print(json.dumps(output, indent=4))
