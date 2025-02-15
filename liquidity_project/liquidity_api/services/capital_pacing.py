#!/usr/bin/env python3
"""
capital_pacing_optimization.py

This script dynamically integrates data from your centralized SQL database and performs
capital pacing optimization using a risk-adjusted return objective. It includes:
  - Dynamic data integration & forecasting from the database.
  - A Sharpe-like objective function for risk-adjusted return.
  - Liquidity, diversification, and minimum investment constraints.
  - Sensitivity analysis on the liquidity buffer.
  - Scenario analysis on the risk-free rate (calibrated from macroeconomic data).
  - Detailed logging & error handling (logs to console and file).
  - Visualization of optimal allocations.
  - JSON output for Django API integration.

Ensure that the centralized database file is available at the location defined in utils/db_utils.py.
"""

import os
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import the centralized DB connection and logging utilities.
from utils.db_utils import get_connection, store_dataframe, DB_PATH  # DB_PATH is defined in the utils
from utils.logging_utils import setup_logging, log_info, log_error
import logging

# Set up logging (both file and console).
setup_logging()
# Add a console handler for immediate log output.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def calibrate_risk_free_rate():
    """
    Calibrates the risk-free rate using the latest 10Y Treasury Yield from the macroeconomic_data table.
    If the value is not found, defaults to 0.02.
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
            # Assume the yield is in percentage terms and convert to decimal.
            risk_free = float(row[0]) / 100.0
            log_info(f"Calibrated risk-free rate from macroeconomic_data: {risk_free}")
            return risk_free
        else:
            log_info("Risk-free rate not found in macroeconomic_data; defaulting to 0.02")
            return 0.02
    except Exception as e:
        log_error(f"Error calibrating risk-free rate: {e}")
        return 0.02

def fetch_strategy_data():
    """
    Fetch strategy-level data from the master_data table.
    For each distinct symbol, calculate:
      - expected_return: Annualized average "Daily Return" (as a proxy for expected return)
      - volatility: Average "Volatility_30"
      - max_commit: Maximum totalInvestments (as the upper bound for commitment)
    Returns:
      expected_returns (np.array), max_commit (np.array), volatility (np.array), symbols (list)
    """
    # Check that the centralized DB file exists.
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT
              symbol,
              AVG("Daily Return") as avg_return,
              AVG("Volatility_30") as avg_volatility,
              MAX(totalInvestments) as max_commit
            FROM master_data
            GROUP BY symbol;
        '''
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            raise ValueError("No strategy data found in master_data table.")
        symbols = []
        expected_returns = []
        volatility = []
        max_commit = []
        for row in rows:
            symbol, avg_return, avg_vol, max_c = row
            symbols.append(symbol)
            # Annualize the daily return (assuming 252 trading days)
            annualized_return = avg_return * 252 if avg_return is not None else 0
            expected_returns.append(annualized_return)
            # Ensure volatility is non-zero; default to 1 if missing or zero.
            vol = avg_vol if (avg_vol is not None and avg_vol > 0) else 1
            volatility.append(vol)
            max_commit.append(max_c if max_c is not None else 0)
        log_info(f"Fetched data for {len(symbols)} strategies.")
        log_info(f"Expected Returns: {expected_returns}")
        log_info(f"Volatility: {volatility}")
        log_info(f"Max Commitments: {max_commit}")
        return (np.array(expected_returns), np.array(max_commit), np.array(volatility), symbols)
    except Exception as e:
        log_error(f"Error fetching strategy data: {e}")
        raise

def fetch_liquidity_data():
    """
    Fetch liquidity data from master_data.
    We use the most recent cashAndCashEquivalents value as our available liquidity.
    Returns:
      initial_liquidity (float), buffer_percentage (float)
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = '''
            SELECT cashAndCashEquivalents 
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
    Forecast capital calls by taking the average of the capital_calls column from master_data.
    This is a placeholder; in a real model, you might use a time series or regression model.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = 'SELECT AVG(capital_calls) FROM master_data;'
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        forecasted_calls = float(row[0]) if row and row[0] is not None else 10_000_000
        log_info(f"Forecasted capital calls: {forecasted_calls}")
        return forecasted_calls
    except Exception as e:
        log_error(f"Error forecasting capital calls: {e}")
        raise

def total_risk_adjusted_return(commitments, expected_returns, volatility, risk_free_rate):
    """
    Objective function: maximize risk-adjusted return (using a Sharpe-like ratio).
    Computes:
      Sharpe_i = (expected_return_i - risk_free_rate) / volatility_i
    and returns the negative sum (since we minimize).
    """
    sharpe_ratios = (expected_returns - risk_free_rate) / volatility
    total = np.sum(commitments * sharpe_ratios)
    return -total

def liquidity_constraint(commitments, initial_liquidity, buffer_percentage, forecasted_calls):
    """
    Constraint to ensure total commitments do not exceed available liquidity.
    Available liquidity = initial_liquidity - forecasted_calls - (buffer_percentage * initial_liquidity)
    """
    buffer = initial_liquidity * buffer_percentage
    max_available = initial_liquidity - forecasted_calls - buffer
    return max_available - np.sum(commitments)

def diversification_constraint(commitments, max_commit):
    """
    Constraint ensuring no single investable strategy exceeds 50% of total commitments.
    Enforced only if at least two strategies are investable.
    """
    investable = np.array(max_commit) > 0
    if np.sum(investable) < 2:
        return 0.0
    total_commitment = np.sum(commitments)
    return np.min(0.5 * total_commitment - commitments[investable])

def minimum_investment_constraint(commitments, min_total_commitment):
    """
    Constraint to ensure that total commitments are at least a minimum fraction of available liquidity.
    """
    return np.sum(commitments) - min_total_commitment

def optimize_commitments():
    """
    Main optimization function that:
      - Pulls dynamic data (strategy parameters, liquidity, forecasted calls, calibrated risk-free rate)
      - Runs constrained optimization to maximize risk-adjusted return
      - Returns results along with strategy symbols.
    """
    try:
        expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
        initial_liquidity, buffer_percentage = fetch_liquidity_data()
        forecasted_calls = forecast_capital_calls()
        risk_free_rate = calibrate_risk_free_rate()
        num_strategies = len(expected_returns)
        x0 = np.zeros(num_strategies)
        # Minimum total investment: 5% of available liquidity.
        min_total_commitment = 0.05 * initial_liquidity

        constraints = [
            {"type": "ineq", "fun": lambda x: liquidity_constraint(x, initial_liquidity, buffer_percentage, forecasted_calls)},
            {"type": "ineq", "fun": lambda x: diversification_constraint(x, max_commit)},
            {"type": "ineq", "fun": lambda x: minimum_investment_constraint(x, min_total_commitment)}
        ]
        bounds = [(0, mc) for mc in max_commit]
        log_info("Starting optimization using dynamic centralized data.")
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
                "risk_free_rate": risk_free_rate
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
    Performs optimization for a range of liquidity buffer percentages.
    Returns a dictionary keyed by buffer percentage.
    """
    results = {}
    for bp in buffer_percentages:
        try:
            expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
            initial_liquidity, _ = fetch_liquidity_data()  # Get liquidity, then override buffer
            forecasted_calls = forecast_capital_calls()
            risk_free_rate = calibrate_risk_free_rate()
            num_strategies = len(expected_returns)
            x0 = np.zeros(num_strategies)
            min_total_commitment = 0.05 * initial_liquidity
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
    Performs optimization for a range of risk-free rate scenarios.
    Returns a dictionary keyed by risk_free_rate value.
    """
    results = {}
    for r in risk_free_rates:
        try:
            expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
            initial_liquidity, buffer_percentage = fetch_liquidity_data()
            forecasted_calls = forecast_capital_calls()
            num_strategies = len(expected_returns)
            x0 = np.zeros(num_strategies)
            min_total_commitment = 0.05 * initial_liquidity
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

def plot_results(opt_result):
    """
    Visualize the optimal capital commitments by strategy using a bar chart.
    """
    try:
        commitments = opt_result.get("optimal_commitments", [])
        symbols = opt_result.get("symbols", [])
        if commitments and symbols:
            plt.figure(figsize=(10, 6))
            plt.bar(symbols, commitments, color='skyblue')
            plt.xlabel('Strategy (Symbol)')
            plt.ylabel('Optimal Commitment')
            plt.title('Optimal Capital Commitments by Strategy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            log_info("No commitments or symbols available for plotting.")
    except Exception as e:
        log_error(f"Error plotting results: {e}")

def run():
    """
    Execute the optimization, perform sensitivity analysis (on liquidity buffer and risk-free rate),
    visualize results, and return a JSON string with the structured output.
    """
    base_result = optimize_commitments()
    sensitivity = run_sensitivity_analysis([0.25, 0.30, 0.35])
    scenarios = run_scenario_analysis([0.01, 0.02, 0.03])
    base_result["sensitivity_analysis"] = sensitivity
    base_result["scenario_analysis"] = scenarios
    plot_results(base_result)
    return json.dumps(base_result, indent=4)

if __name__ == "__main__":
    print(run())
