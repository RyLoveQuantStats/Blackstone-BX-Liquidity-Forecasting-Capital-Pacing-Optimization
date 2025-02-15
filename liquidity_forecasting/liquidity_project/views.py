from django.shortcuts import render

# liquidity_api/views.py

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

# Import the run() functions from your services scripts.
from liquidity_forecasting.liquidity_project.models.captial_pacing.model import run as run_capital_pacing
from liquidity_forecasting.liquidity_project.models.liqudity_forecast.model import run as run_liquidity_forecast
from liquidity_forecasting.liquidity_project.models.liqudity_forecast.monte_liquidity import run as run_monte_carlo

@require_http_methods(["GET"])
def capital_pacing_api(request):
    result = run_capital_pacing()
    return JsonResponse(result)

@require_http_methods(["GET"])
def liquidity_forecast_api(request):
    result = run_liquidity_forecast()
    return JsonResponse(result)

@require_http_methods(["GET"])
def monte_carlo_api(request):
    result = run_monte_carlo()
    return JsonResponse(result)

