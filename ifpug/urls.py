from django.urls import path
from . import views


urlpatterns = [
    path('', views.ifpug, name='ifpug'),
]
