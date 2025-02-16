from django.shortcuts import render

# models_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import argparse

# Import your functions from model.py
from models_api.model import run_forecasting, run_simulation

class SarimaxForecastAPIView(APIView):
    """
    API endpoint for running the SARIMAX-only forecasting model.
    """
    def get(self, request, format=None):
        output = run_forecasting()
        return Response(output, status=status.HTTP_200_OK)

class CombinedModelAPIView(APIView):
    """
    API endpoint for running the combined model (SARIMAX + Rolling Exp Smoothing + Monte Carlo).
    Accepts simulation parameters via query parameters.
    """
    def get(self, request, format=None):
        # Retrieve simulation parameters from query parameters, with defaults
        n_simulations = int(request.query_params.get("n_simulations", 100000))
        horizon = int(request.query_params.get("horizon", 30))
        method = request.query_params.get("method", "normal")
        stress_mean = float(request.query_params.get("stress_mean", 1.0))
        stress_std = float(request.query_params.get("stress_std", 1.0))
        block_size = int(request.query_params.get("block_size", 5))
        rolling_window = int(request.query_params.get("rolling_window", 90))
        parallel = request.query_params.get("parallel", "false").lower() == "true"
        stress_mean_list = request.query_params.get("stress_mean_list", "1.0")
        stress_std_list = request.query_params.get("stress_std_list", "1.0")
        
        # Create an argparse.Namespace to mimic command-line arguments
        args = argparse.Namespace(
            n_simulations=n_simulations,
            horizon=horizon,
            method=method,
            stress_mean=stress_mean,
            stress_std=stress_std,
            block_size=block_size,
            rolling_window=rolling_window,
            parallel=parallel,
            stress_mean_list=stress_mean_list,
            stress_std_list=stress_std_list
        )
        output = run_simulation(args)
        return Response(output, status=status.HTTP_200_OK)

