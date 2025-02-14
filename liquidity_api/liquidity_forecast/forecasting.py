from django.http import JsonResponse
from rest_framework.decorators import api_view
from .forecasting import run_capital_pacing, run_liquidity_forecast, run_monte_carlo

@api_view(['GET'])
def capital_pacing_api(request):
    result = run_capital_pacing()
    return JsonResponse({'capital_pacing': result})

@api_view(['GET'])
def liquidity_forecast_api(request):
    result = run_liquidity_forecast()
    return JsonResponse({'liquidity_forecast': result})

@api_view(['GET'])
def monte_carlo_api(request):
    result = run_monte_carlo()
    return JsonResponse({'monte_carlo': result})

# In forecasting.py, wrap each model script into a function:
# Example:
# def run_capital_pacing():
#     # Paste your capital pacing model code here
#     # Return the output

# Add the views to urls.py
# urlpatterns = [
#     path('api/capital_pacing/', capital_pacing_api, name='capital_pacing_api'),
#     path('api/liquidity_forecast/', liquidity_forecast_api, name='liquidity_forecast_api'),
#     path('api/monte_carlo/', monte_carlo_api, name='monte_carlo_api'),
# ]
