from django.urls import path
from .views import ClassificationAPI

urlpatterns = [
   path('', ClassificationAPI.as_view(), name="api_views"),
]