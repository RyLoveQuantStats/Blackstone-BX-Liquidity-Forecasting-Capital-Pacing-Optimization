#!/usr/bin/env python3

"""
capital_pacing_optimization.py

A simple example using scipy.optimize to handle a single-period problem:
we forecast some 'capital_calls' next period, and try to commit capital 
to 3 strategies without breaching liquidity constraints.
"""

import numpy as np
from scipy.optimize import minimize

# Hypothetical scenario:
expected_returns = np.array([0.12, 0.10, 0.15])
initial_liquidity = 100_000_000
max_commit = np.array([50_000_000, 60_000_000, 40_000_000])

def total_returns(commitments):
    """Use negative sign so we can maximize by 'minimize' function."""
    return -np.sum(commitments * expected_returns)

def liquidity_constraint(commitments):
    """
    Suppose we forecast capital_calls at 10M (e.g. from ARIMA or Monte Carlo).
    We also maintain a 30% liquidity buffer. 
    => max available = initial_liquidity - capital_calls - buffer
    """
    forecasted_calls = 10_000_000
    buffer = initial_liquidity * 0.30
    max_available = initial_liquidity - forecasted_calls - buffer
    return max_available - np.sum(commitments)

def main():
    cons = ({ "type": "ineq", "fun": liquidity_constraint },)
    bounds = [(0, mc) for mc in max_commit]
    x0 = np.zeros(len(expected_returns))

    solution = minimize(total_returns, x0, 
                        constraints=cons, 
                        bounds=bounds, 
                        method="SLSQP")
    if solution.success:
        optimal_commitments = solution.x
        print("Optimal commitments:", optimal_commitments)
        print("Max Return Achieved:", -solution.fun)
    else:
        print("Optimization failed:", solution.message)

if __name__ == "__main__":
    main()
