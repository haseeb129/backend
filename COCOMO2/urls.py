from django.urls import path
from . import views


urlpatterns = [
    path('', views.COCOMO2, name='COCOMO2')
]
