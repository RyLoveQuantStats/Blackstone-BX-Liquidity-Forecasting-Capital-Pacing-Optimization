#!/usr/bin/env python
import argparse
import json

# Import your forecasting & simulation functions
from liquidity_project.models.calls_SARIMAX_Monte.model import run_forecasting, run_simulation

def parse_args():
    """
    Parses command‑line arguments.
    The --mode argument selects either "forecast" or "simulation".
    Simulation mode requires additional parameters.
    """
    parser = argparse.ArgumentParser(description="Liquidity Forecasting and Simulation Model")
    parser.add_argument("--mode", type=str, default="forecast", choices=["forecast", "simulation"],
                        help="Select mode: forecast or simulation")
    # Simulation-specific parameters:
    parser.add_argument("--n_simulations", type=int, default=100000,
                        help="Number of simulations (simulation mode)")
    parser.add_argument("--horizon", type=int, default=30,
                        help="Forecast horizon in days (simulation mode)")
    parser.add_argument("--method", type=str, default="normal",
                        choices=["normal", "t", "bootstrap", "kde", "block", "rolling", "garch"],
                        help="Method to simulate daily shocks (simulation mode)")
    parser.add_argument("--stress_mean", type=float, default=1.0,
                        help="Stress multiplier for mean (simulation mode)")
    parser.add_argument("--stress_std", type=float, default=1.0,
                        help="Stress multiplier for std deviation (simulation mode)")
    parser.add_argument("--block_size", type=int, default=5,
                        help="Block size for block bootstrapping (simulation mode)")
    parser.add_argument("--rolling_window", type=int, default=90,
                        help="Window size for rolling simulation (simulation mode)")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing (simulation mode)")
    parser.add_argument("--stress_mean_list", type=str, default="1.0",
                        help="Comma-separated list of stress mean factors (simulation mode)")
    parser.add_argument("--stress_std_list", type=str, default="1.0",
                        help="Comma-separated list of stress std factors (simulation mode)")
    return parser.parse_args()

def convert_timestamp_keys_to_str(data):
    """
    Recursively converts any pandas Timestamp keys in a nested dict structure to strings.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # Convert the key to string if it's not a standard type
            new_key = str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
            new_dict[new_key] = convert_timestamp_keys_to_str(v)
        return new_dict
    elif isinstance(data, list):
        return [convert_timestamp_keys_to_str(item) for item in data]
    else:
        return data

def main():
    args = parse_args()

    if args.mode == "forecast":
        output = run_forecasting()
    elif args.mode == "simulation":
        output = run_simulation(args)
    else:
        output = {"error": "Invalid mode selected."}

    # Convert any Timestamp keys to string for JSON serialization
    output_str_keys = convert_timestamp_keys_to_str(output)

    print(json.dumps(output_str_keys, indent=4))

if __name__ == "__main__":
    main()
