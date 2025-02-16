"""
main.py
-------
This file serves as the entry point for executing the capital pacing optimization.
It imports the model functions, runs the optimization, performs sensitivity and scenario analyses,
and prints the JSON results.
"""

import json
import pandas as pd
from liquidity_project.models.capital_pacing.model import (
    optimize_commitments,
    plot_capital_calls_forecast,
    plot_optimal_commitments,
    run_sensitivity_and_scenario
)

def run():
    """
    Executes the optimization, performs sensitivity and scenario analyses,
    visualizes forecast and optimal allocations, and returns a JSON string with results.
    """
    base_result = optimize_commitments()

    # Plot the forecast if available.
    if "capital_calls_history" in base_result and base_result["capital_calls_history"] is not None:
        history_series = pd.Series(base_result["capital_calls_history"])
        forecast_series = pd.Series(base_result["capital_calls_forecast"])
        if base_result["capital_calls_conf_int"] is not None:
            conf_df = pd.DataFrame(base_result["capital_calls_conf_int"])
        else:
            conf_df = None
        plot_capital_calls_forecast(history_series, forecast_series, conf_df)

    # Plot optimal commitments if available.
    if "error" not in base_result:
        plot_optimal_commitments(base_result)

    sensitivity_results = run_sensitivity_and_scenario()
    base_result.update(sensitivity_results)

    return json.dumps(base_result, indent=4)

if __name__ == "__main__":
    print(run())
