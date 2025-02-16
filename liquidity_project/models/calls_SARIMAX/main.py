"""
main.py
-------
This file serves as the entry point for executing the capital pacing optimization.
It imports the model functions, runs the optimization, performs sensitivity and scenario analyses,
and prints the JSON results.
"""

import argparse
import json
from liquidity_project.models.calls_SARIMAX import run_forecasting

def parse_args():
    parser = argparse.ArgumentParser(description="SARIMAX Forecasting Model")
    # (Add any arguments you need; for example, if run_forecasting has parameters)
    return parser.parse_args()

def main():
    args = parse_args()
    # Here we simply run the forecasting routine.
    output = run_forecasting()
    
    # If your output dictionary contains pandas Timestamps as keys,
    # convert them to strings for JSON serialization:
    def convert_keys(d):
        if isinstance(d, dict):
            return {str(k): convert_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_keys(item) for item in d]
        else:
            return d
    output = convert_keys(output)
    
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()