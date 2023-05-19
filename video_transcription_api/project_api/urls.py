from django.urls import path
from .views import TranscribeAPI

urlpatterns = [
   path('', TranscribeAPI.as_view(), name="api_views"),
]
