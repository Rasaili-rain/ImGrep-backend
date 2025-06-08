from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check),
    path('test/', views.test_api),
]