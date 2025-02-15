# liquidity_api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('capital-pacing/', views.capital_pacing_api, name='capital_pacing_api'),
    path('liquidity-forecast/', views.liquidity_forecast_api, name='liquidity_forecast_api'),
    path('monte-carlo/', views.monte_carlo_api, name='monte_carlo_api'),
]


