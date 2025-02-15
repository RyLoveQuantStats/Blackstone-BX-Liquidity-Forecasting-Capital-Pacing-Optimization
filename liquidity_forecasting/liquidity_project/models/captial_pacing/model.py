"""
Capital Pacing Optimization Script with SARIMAX Forecast
--------------------------------------------------------
This script dynamically integrates data from the SQL database and performs
capital pacing optimization using a risk-adjusted return objective. It includes:
  - Dynamic data integration from the database.
  - SARIMAX-based forecasting of capital calls (with confidence intervals).
  - A Sharpe-like objective function for risk-adjusted return.
  - Liquidity, diversification, and minimum investment constraints.
  - Sensitivity analysis on the liquidity buffer.
  - Scenario analysis on the risk-free rate (calibrated from macroeconomic data).
  - Detailed logging & error handling (logs to console and file).
  - Visualization of both the capital calls forecast and optimal allocations.
  - JSON output for Django API integration.
"""

import os
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt

# If you want auto_arima, uncomment:
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scipy.optimize import minimize

# Import the centralized DB connection and logging utilities.
from utils.db_utils import get_connection, store_dataframe, DB_PATH  # DB_PATH is defined in the utils
from utils.logging_utils import setup_logging, log_info, log_error
import logging

# Set up logging (both file and console).
setup_logging()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def calibrate_risk_free_rate():
    """
    Calibrates the risk-free rate using the latest 10Y Treasury Yield from the macroeconomic_data table.
    If the value is not found, defaults to 0.02 (2%).
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

            # Annualize the daily return (assuming 252 trading days per year)
            if avg_return is None:
                annualized_return = 0.0
            else:
                annualized_return = avg_return * 252
            expected_returns.append(annualized_return)

            # Ensure volatility is non-zero; default to 1 if missing or zero.
            if avg_vol is None or avg_vol <= 0:
                vol = 1.0
            else:
                vol = avg_vol
            volatility.append(vol)

            if max_c is None:
                max_commit.append(0.0)
            else:
                max_commit.append(max_c)

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
    Forecast capital calls using a SARIMAX model.
    Returns a single float representing the sum of the next 12 months' forecasts.
    Also returns the forecast series and confidence intervals for plotting.
    """
    import pandas as pd  # local import for clarity

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

        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=['Date', 'capital_calls'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        if df['capital_calls'].isnull().all():
            log_info("All capital_calls are null; defaulting to 10,000,000.")
            return 10_000_000, None, None, None

        # Simple fill for missing data (forward fill). Adjust as needed.
        df['capital_calls'] = df['capital_calls'].fillna(method='ffill')

        # Optionally, you can use auto_arima to find the best order:
        # auto_arima_model = auto_arima(df['capital_calls'], seasonal=True, m=12)
        # order = auto_arima_model.order
        # seasonal_order = auto_arima_model.seasonal_order

        # Manual SARIMAX order (p, d, q) = (1, 1, 1), seasonal (P, D, Q, m) = (1, 1, 1, 12)
        model = SARIMAX(
            df['capital_calls'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # Forecast next 12 months
        forecast_steps = 12
        forecast_obj = results.get_forecast(steps=forecast_steps)
        forecast_values = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        # Sum of next 12 months
        forecasted_calls = float(forecast_values.sum())

        log_info(f"Forecasted capital calls (12-month sum): {forecasted_calls}")
        return forecasted_calls, df['capital_calls'], forecast_values, conf_int

    except Exception as e:
        log_error(f"Error forecasting capital calls: {e}")
        return 10_000_000, None, None, None


def plot_capital_calls_forecast(history, forecast_values, conf_int):
    """
    Plot historical capital calls and forecast with confidence intervals.
    """
    if history is None or forecast_values is None:
        log_info("Insufficient data for capital calls forecast plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history, label="Historical Capital Calls", color="blue")

    # Extend the index for the forecast
    forecast_index = forecast_values.index
    plt.plot(forecast_index, forecast_values, label="Forecasted Capital Calls", color="green")

    if conf_int is not None:
        plt.fill_between(
            forecast_index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='gray',
            alpha=0.3,
            label="Confidence Interval"
        )

    plt.title("Capital Calls Forecast (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Capital Calls")
    plt.legend()
    plt.tight_layout()
    plt.show()


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
        # If there's only one strategy or none, no constraint.
        return 0.0
    total_commitment = np.sum(commitments)
    # The smallest value of (0.5 * total_commitment - commitments[i]) must be >= 0
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

        # Forecast capital calls using SARIMAX
        forecasted_calls, hist_calls, forecast_vals, conf_int = forecast_capital_calls()

        risk_free_rate = calibrate_risk_free_rate()
        num_strategies = len(expected_returns)

        # Initial guess: start with zero commitments
        x0 = np.zeros(num_strategies)

        # Minimum total investment: 5% of available liquidity
        min_total_commitment = 0.05 * initial_liquidity

        # Constraints
        constraints = [
            {"type": "ineq", "fun": lambda x: liquidity_constraint(x, initial_liquidity, buffer_percentage, forecasted_calls)},
            {"type": "ineq", "fun": lambda x: diversification_constraint(x, max_commit)},
            {"type": "ineq", "fun": lambda x: minimum_investment_constraint(x, min_total_commitment)}
        ]

        # Bounds: each strategy from 0 up to max_commit
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
            max_return = float(-solution.fun)  # since we negated the Sharpe sum
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
                # We'll store the historical & forecast data indices for plotting
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
    Performs optimization for a range of liquidity buffer percentages.
    Returns a dictionary keyed by buffer percentage.
    """
    results = {}
    for bp in buffer_percentages:
        try:
            expected_returns, max_commit, volatility, symbols = fetch_strategy_data()
            initial_liquidity, _ = fetch_liquidity_data()  # Get liquidity, then override buffer
            forecasted_calls, _, _, _ = forecast_capital_calls()
            risk_free_rate = calibrate_risk_free_rate()
            num_strategies = len(expected_returns)
            x0 = np.zeros(num_strategies)

            min_total_commitment = 0.05 * initial_liquidity

            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x, bp=bp: liquidity_constraint(x, initial_liquidity, bp, forecasted_calls)
                },
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
            forecasted_calls, _, _, _ = forecast_capital_calls()
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


def plot_optimal_commitments(opt_result):
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
    visualize results (both the forecast and optimal allocations), and return a JSON string with the structured output.
    """
    base_result = optimize_commitments()

    # Plot the forecast
    if "capital_calls_history" in base_result and base_result["capital_calls_history"] is not None:
        import pandas as pd
        history_series = pd.Series(base_result["capital_calls_history"])
        forecast_series = pd.Series(base_result["capital_calls_forecast"])
        from pandas import DataFrame
        if base_result["capital_calls_conf_int"] is not None:
            conf_df = DataFrame(base_result["capital_calls_conf_int"])
        else:
            conf_df = None
        plot_capital_calls_forecast(history_series, forecast_series, conf_df)

    # Plot the optimal commitments
    if "error" not in base_result:
        plot_optimal_commitments(base_result)

    # Sensitivity analysis (liquidity buffer)
    buffer_sensitivity = run_sensitivity_analysis([0.25, 0.30, 0.35])

    # Scenario analysis (risk-free rates)
    rf_scenario = run_scenario_analysis([0.01, 0.02, 0.03])

    # Attach them to the base result
    base_result["sensitivity_analysis"] = buffer_sensitivity
    base_result["scenario_analysis"] = rf_scenario

    # Return JSON string
    return json.dumps(base_result, indent=4)


if __name__ == "__main__":
    print(run())
