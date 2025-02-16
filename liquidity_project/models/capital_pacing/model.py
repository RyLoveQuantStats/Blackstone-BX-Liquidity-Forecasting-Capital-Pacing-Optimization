"""
model.py
--------
This module contains the data integration, forecasting, optimization,
and plotting functions for capital pacing optimization with SARIMAX forecasting.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import logging

# Uncomment if using auto_arima:
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import minimize

# Import DB and logging utilities.
from utils.db_utils import get_connection, store_dataframe, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error

# Set up logging (both file and console).
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# ---------------------------
# 1. Data Fetching & Forecasting Functions
# ---------------------------

def calibrate_risk_free_rate():
    """
    Calibrates the risk-free rate using the latest 10Y Treasury Yield from the macroeconomic_data table.
    If not found, defaults to 0.02 (2%).
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT "10Y Treasury Yield"
            FROM macroeconomic_data
            ORDER BY "index" DESC
            LIMIT 1;
        '''
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        if row and row[0] is not None:
            risk_free = float(row[0]) / 100.0
            log_info(f"Calibrated risk-free rate from macroeconomic_data: {risk_free}")
            return risk_free
        else:
            log_info("Risk-free rate not found; defaulting to 0.02")
            return 0.02
    except Exception as e:
        log_error(f"Error calibrating risk-free rate: {e}")
        return 0.02

def fetch_strategy_data():
    """
    Fetches overall strategy-level data from the master_data table for a single ticker.
    Computes:
      - expected_return: Annualized average "Daily Return"
      - volatility: Average "Volatility_30"
      - max_commit: Maximum of 'fin_total_assets' (as a proxy for available commitment)
    
    Returns:
      expected_returns (np.array), max_commit (np.array), volatility (np.array), symbols (list)
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT 
              AVG("Daily Return") as avg_return,
              AVG("Volatility_30") as avg_volatility,
              MAX("fin_total_assets") as max_commit
            FROM master_data;
        '''
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError("No strategy data found in master_data table.")
        
        avg_return, avg_vol, max_c = row
        # Annualize the daily return (assuming 252 trading days)
        annualized_return = (avg_return * 252) if avg_return is not None else 0.0
        vol = avg_vol if (avg_vol is not None and avg_vol > 0) else 1.0
        max_commit_val = max_c if max_c is not None else 0.0

        # For now, assume one ticker.
        symbol = "KKR"  # Replace as needed.
        log_info(f"Fetched strategy data: expected_return = {annualized_return}, volatility = {vol}, max_commit = {max_commit_val}")
        return np.array([annualized_return]), np.array([max_commit_val]), np.array([vol]), [symbol]
    except Exception as e:
        log_error(f"Error fetching strategy data: {e}")
        raise

def fetch_liquidity_data():
    """
    Fetches liquidity data from master_data.
    Uses the most recent value from the 'fin_cash_and_cash_equivalents' column as available liquidity.
    Returns:
      initial_liquidity (float), buffer_percentage (float)
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT fin_cash_and_cash_equivalents
            FROM master_data
            ORDER BY Date DESC
            LIMIT 1;
        '''
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] is None:
            raise ValueError("No liquidity data found in master_data table.")

        initial_liquidity = float(row[0])
        # Default liquidity buffer percentage (reserve 30% of liquidity)
        buffer_percentage = 0.30

        log_info(f"Initial liquidity: {initial_liquidity}, using buffer percentage: {buffer_percentage}")
        return initial_liquidity, buffer_percentage

    except Exception as e:
        log_error(f"Error fetching liquidity data: {e}")
        raise

def forecast_capital_calls():
    """
    Forecasts capital calls using a SARIMAX model.
    Returns the sum of the next 12 months' forecasts, as well as the historical series,
    forecast series, and confidence intervals.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT Date, capital_calls
            FROM master_data
            WHERE capital_calls IS NOT NULL
            ORDER BY Date ASC;
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            log_info("No capital calls data found; defaulting to 10,000,000.")
            return 10_000_000, None, None, None

        df = pd.DataFrame(rows, columns=['Date', 'capital_calls'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        # Use the updated method to fill missing values.
        df['capital_calls'] = df['capital_calls'].ffill()
        # Set a frequency (e.g., monthly) to help SARIMAX; adjust as appropriate.
        if df.index.inferred_freq is None:
            df = df.asfreq('MS')

        # Use manual SARIMAX order.
        model = SARIMAX(df['capital_calls'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast_steps = 12
        forecast_obj = results.get_forecast(steps=forecast_steps)
        forecast_values = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        forecasted_calls = float(forecast_values.sum())
        log_info(f"Forecasted capital calls (12-month sum): {forecasted_calls}")
        return forecasted_calls, df['capital_calls'], forecast_values, conf_int
    except Exception as e:
        log_error(f"Error forecasting capital calls: {e}")
        return 10_000_000, None, None, None

def plot_capital_calls_forecast(history, forecast_values, conf_int):
    """
    Plots historical capital calls and the forecast with confidence intervals.
    """
    if history is None or forecast_values is None:
        log_info("Insufficient data for capital calls forecast plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history, label="Historical Capital Calls", color="blue")
    forecast_index = forecast_values.index
    plt.plot(forecast_index, forecast_values, label="Forecasted Capital Calls", color="green")
    if conf_int is not None:
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                         color='gray', alpha=0.3, label="Confidence Interval")
    plt.title("Capital Calls Forecast (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Capital Calls")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 2. Optimization Functions
# ---------------------------

def total_risk_adjusted_return(commitments, expected_returns, volatility, risk_free_rate):
    """
    Objective function: maximize risk-adjusted return using a Sharpe-like ratio.
    Computes Sharpe_i = (expected_return_i - risk_free_rate) / volatility_i and returns negative sum.
    """
    sharpe_ratios = (expected_returns - risk_free_rate) / volatility
    total = np.sum(commitments * sharpe_ratios)
    return -total

def liquidity_constraint(commitments, initial_liquidity, buffer_percentage, forecasted_calls):
    """
    Ensures total commitments do not exceed available liquidity.
    """
    buffer = initial_liquidity * buffer_percentage
    max_available = initial_liquidity - forecasted_calls - buffer
    return max_available - np.sum(commitments)

def diversification_constraint(commitments, max_commit):
    """
    Ensures no single strategy exceeds 50% of total commitments.
    """
    investable = np.array(max_commit) > 0
    if np.sum(investable) < 2:
        return 0.0
    total_commitment = np.sum(commitments)
    return np.min(0.5 * total_commitment - commitments[investable])

def minimum_investment_constraint(commitments, min_total_commitment):
    """
    Ensures total commitments are at least a minimum fraction of available liquidity.
    """
    return np.sum(commitments) - min_total_commitment

def optimize_commitments():
    """
    Main optimization function:
      - Fetches strategy data, liquidity, forecasted capital calls, and calibrated risk-free rate.
      - Runs constrained optimization to maximize risk-adjusted return.
      - Returns a result dictionary.
    """
    try:
        # Fetch data once for optimization.
        expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
        initial_liquidity, buffer_percentage = fetch_liquidity_data()
        forecasted_calls, hist_calls, forecast_vals, conf_int = forecast_capital_calls()
        risk_free_rate = calibrate_risk_free_rate()
        num_strategies = len(expected_returns)
        
        # Use a more informed initial guess: a small fraction of max_commit.
        x0 = 0.05 * max_commit  
        min_total_commitment = 0.05 * initial_liquidity

        constraints = [
            {"type": "ineq", "fun": lambda x: liquidity_constraint(x, initial_liquidity, buffer_percentage, forecasted_calls)},
            {"type": "ineq", "fun": lambda x: diversification_constraint(x, max_commit)},
            {"type": "ineq", "fun": lambda x: minimum_investment_constraint(x, min_total_commitment)}
        ]
        bounds = [(0, mc) for mc in max_commit]

        log_info("Starting optimization using dynamic data.")
        solution = minimize(
            total_risk_adjusted_return,
            x0,
            args=(expected_returns, volatility, risk_free_rate),
            constraints=constraints,
            bounds=bounds,
            method="SLSQP"
        )

        if solution.success:
            optimal_commitments = solution.x
            max_return = float(-solution.fun)
            log_info("Optimization successful.")
            result = {
                "symbols": symbols,
                "optimal_commitments": optimal_commitments.tolist(),
                "max_risk_adjusted_return": max_return,
                "initial_liquidity": initial_liquidity,
                "forecasted_calls": forecasted_calls,
                "buffer_percentage": buffer_percentage,
                "min_total_commitment": min_total_commitment,
                "risk_free_rate": risk_free_rate,
                "capital_calls_history": hist_calls.to_dict() if hist_calls is not None else None,
                "capital_calls_forecast": forecast_vals.to_dict() if forecast_vals is not None else None,
                "capital_calls_conf_int": conf_int.to_dict() if conf_int is not None else None
            }
            return result
        else:
            error_msg = solution.message
            log_error(f"Optimization failed: {error_msg}")
            return {"error": error_msg}
    except Exception as e:
        log_error(f"An error occurred during optimization: {e}")
        return {"error": str(e)}

def run_sensitivity_analysis(buffer_percentages):
    """
    Performs optimization over a range of liquidity buffer percentages.
    """
    results = {}
    # Fetch the common data only once.
    expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
    initial_liquidity, _ = fetch_liquidity_data()
    forecasted_calls, _, _, _ = forecast_capital_calls()
    risk_free_rate = calibrate_risk_free_rate()
    num_strategies = len(expected_returns)
    min_total_commitment = 0.05 * initial_liquidity
    for bp in buffer_percentages:
        try:
            x0 = 0.05 * max_commit  # More informed initial guess.
            constraints = [
                {"type": "ineq", "fun": lambda x, bp=bp: liquidity_constraint(x, initial_liquidity, bp, forecasted_calls)},
                {"type": "ineq", "fun": lambda x: diversification_constraint(x, max_commit)},
                {"type": "ineq", "fun": lambda x: minimum_investment_constraint(x, min_total_commitment)}
            ]
            bounds = [(0, mc) for mc in max_commit]
            solution = minimize(
                total_risk_adjusted_return,
                x0,
                args=(expected_returns, volatility, risk_free_rate),
                constraints=constraints,
                bounds=bounds,
                method="SLSQP"
            )
            if solution.success:
                results[f"buffer_{bp}"] = {
                    "optimal_commitments": solution.x.tolist(),
                    "max_risk_adjusted_return": float(-solution.fun)
                }
            else:
                results[f"buffer_{bp}"] = {"error": solution.message}
        except Exception as e:
            results[f"buffer_{bp}"] = {"error": str(e)}
    return results

def run_scenario_analysis(risk_free_rates):
    """
    Performs optimization over a range of risk-free rate scenarios.
    """
    results = {}
    # Fetch common data only once.
    expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
    initial_liquidity, buffer_percentage = fetch_liquidity_data()
    forecasted_calls, _, _, _ = forecast_capital_calls()
    num_strategies = len(expected_returns)
    min_total_commitment = 0.05 * initial_liquidity
    for r in risk_free_rates:
        try:
            x0 = 0.05 * max_commit  # More informed initial guess.
            constraints = [
                {"type": "ineq", "fun": lambda x: liquidity_constraint(x, initial_liquidity, buffer_percentage, forecasted_calls)},
                {"type": "ineq", "fun": lambda x: diversification_constraint(x, max_commit)},
                {"type": "ineq", "fun": lambda x: minimum_investment_constraint(x, min_total_commitment)}
            ]
            bounds = [(0, mc) for mc in max_commit]
            solution = minimize(
                total_risk_adjusted_return,
                x0,
                args=(expected_returns, volatility, r),
                constraints=constraints,
                bounds=bounds,
                method="SLSQP"
            )
            if solution.success:
                results[f"risk_free_{r}"] = {
                    "optimal_commitments": solution.x.tolist(),
                    "max_risk_adjusted_return": float(-solution.fun)
                }
            else:
                results[f"risk_free_{r}"] = {"error": solution.message}
        except Exception as e:
            results[f"risk_free_{r}"] = {"error": str(e)}
    return results

def plot_optimal_commitments(opt_result):
    """
    Visualizes the optimal capital commitments by strategy using a bar chart.
    """
    try:
        commitments = opt_result.get("optimal_commitments", [])
        symbols = opt_result.get("symbols", [])
        if commitments and symbols:
            plt.figure(figsize=(10, 6))
            plt.bar(symbols, commitments, color='skyblue')
            plt.xlabel('Strategy (Ticker)')
            plt.ylabel('Optimal Commitment')
            plt.title('Optimal Capital Commitments')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            log_info("No commitments or symbols available for plotting.")
    except Exception as e:
        log_error(f"Error plotting optimal commitments: {e}")

def run_sensitivity_and_scenario():
    """
    Runs both sensitivity analysis (on liquidity buffer) and scenario analysis (on risk-free rate),
    and returns a dictionary with the combined results.
    """
    buffer_sensitivity = run_sensitivity_analysis([0.25, 0.30, 0.35])
    rf_scenario = run_scenario_analysis([0.01, 0.02, 0.03])
    return {"sensitivity_analysis": buffer_sensitivity, "scenario_analysis": rf_scenario}
