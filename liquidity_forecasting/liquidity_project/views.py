from django.shortcuts import render

# liquidity_api/views.py

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

# Import the run() functions from your services scripts.
from liquidity_project.models.capital_pacing import run as run_capital_pacing
from liquidity_project.models.liquidity_forecast import run as run_liquidity_forecast
from liquidity_project.models.simulate_liquidity_monte import run as run_monte_carlo

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

