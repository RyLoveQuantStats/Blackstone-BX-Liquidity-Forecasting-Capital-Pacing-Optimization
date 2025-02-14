from django.http import JsonResponse
from rest_framework.decorators import api_view
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Adjusted DB path using BASE_DIR from settings.py
from django.conf import settings
DB_PATH = os.path.join(settings.BASE_DIR, 'database', 'blackstone_data.db')
TABLE_NAME = "bx_master_data"

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn, parse_dates=["Date"])
        conn.close()
        df.set_index("Date", inplace=True)
        return df
    except sqlite3.OperationalError as e:
        return JsonResponse({"error": f"Database connection failed: {str(e)}"}, status=500)
    print(f"DB_PATH used: {DB_PATH}")


@api_view(['GET'])
def optimization_view(request):
    expected_returns = np.array([0.12, 0.10, 0.15])
    initial_liquidity = 100_000_000
    max_commit = np.array([50_000_000, 60_000_000, 40_000_000])

    def total_returns(commitments):
        return -np.sum(commitments * expected_returns)

    def liquidity_constraint(commitments):
        forecasted_calls = 10_000_000
        buffer = initial_liquidity * 0.30
        return initial_liquidity - forecasted_calls - buffer - np.sum(commitments)

    cons = ({"type": "ineq", "fun": liquidity_constraint},)
    bounds = [(0, mc) for mc in max_commit]
    solution = minimize(total_returns, np.zeros(3), constraints=cons, bounds=bounds, method="SLSQP")

    return JsonResponse({"optimal_commitments": solution.x.tolist(), "max_return": -solution.fun})

@api_view(['GET'])
def forecast_view(request):
    df = load_data()
    if isinstance(df, JsonResponse):
        return df

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    model = SARIMAX(train["capital_calls"], order=(2,1,2)).fit(disp=False)
    forecast = model.forecast(steps=len(test))
    return JsonResponse({"forecast": forecast.tolist()})

@api_view(['GET'])
def montecarlo_view(request):
    df = load_data()
    if isinstance(df, JsonResponse):
        return df

    changes = df["capital_calls"].diff().dropna()
    mean, std = changes.mean(), changes.std()
    last_value = df["capital_calls"].iloc[-1]
    simulations = [max(0, last_value + np.sum(np.random.normal(mean, std, 30))) for _ in range(1000)]
    return JsonResponse({"p5": np.percentile(simulations, 5), "p50": np.percentile(simulations, 50), "p95": np.percentile(simulations, 95)})

