#!/usr/bin/env python3

"""
capital_pacing_optimization.py

Example of using scipy.optimize to decide on new commitments to different
“funds/strategies” while ensuring we retain enough liquidity 
to cover projected 'capital_calls'.

This is a simplified single-period example.
"""

import numpy as np
from scipy.optimize import minimize

# Let's say we have 3 hypothetical PE strategies:
expected_returns = np.array([0.12, 0.10, 0.15])  # annual returns
initial_liquidity = 100_000_000  # $100M
max_commit = np.array([50_000_000, 60_000_000, 40_000_000])

def total_returns(commitments):
    """
    Negative sum of (commitment_i * expected_return_i),
    because we use 'minimize' to find the max.
    """
    return -np.sum(commitments * expected_returns)

def liquidity_constraint(commitments):
    """
    Suppose we forecast, say, 10M in 'capital_calls' next period 
    from the ARIMA or Monte Carlo steps. We want to keep a 30% buffer 
    of total liquidity. 
    So we can only commit up to (initial_liquidity - calls - buffer).
    """
    forecasted_calls = 10_000_000  # e.g. from your ARIMA model or a scenario
    buffer = initial_liquidity * 0.30  # 30% buffer
    max_available = initial_liquidity - forecasted_calls - buffer
    return max_available - np.sum(commitments)

bounds = [(0, max_commit[i]) for i in range(len(max_commit))]

cons = ({ "type": "ineq", "fun": liquidity_constraint },)

def main():
    x0 = np.zeros(len(expected_returns))  # start with 0 commitments

    solution = minimize(
        total_returns,
        x0,
        constraints=cons,
        bounds=bounds,
        method="SLSQP"
    )

    if solution.success:
        optimal_commitments = solution.x
        print("Optimal commitments to each strategy:", optimal_commitments)
        print("Max Return Achieved:", -solution.fun)  # since we negated it
    else:
        print("Optimization failed:", solution.message)

if __name__ == "__main__":
    main()
