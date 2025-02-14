"""
URL configuration for liquidity_api project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# forecast/urls.py
from django.contrib import admin
from django.urls import path
from liquidity_forecast.views import optimization_view, forecast_view, montecarlo_view

from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the Liquidity Forecasting API")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),  # Add this line for the homepage
    path('api/forecast/capitalpacing/', optimization_view, name='capital_pacing'),
    path('api/forecast/arima/', forecast_view, name='arima_forecast'),
    path('api/forecast/montecarlo/', montecarlo_view, name='monte_carlo'),
]


