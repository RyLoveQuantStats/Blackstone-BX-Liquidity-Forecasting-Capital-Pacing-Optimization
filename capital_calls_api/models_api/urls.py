# models_api/urls.py
from django.urls import path
from models_api.views import SarimaxForecastAPIView, CombinedModelAPIView

urlpatterns = [
    path('sarimax/', SarimaxForecastAPIView.as_view(), name='sarimax_forecast'),
    path('combined/', CombinedModelAPIView.as_view(), name='combined_model'),
]
