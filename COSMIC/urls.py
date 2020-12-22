from django.urls import path
from . import views


urlpatterns = [
    path('', views.COSMIC, name='COSMIC'),
]
