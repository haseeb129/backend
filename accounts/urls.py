"""drf URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from .views import RegisterAPI, LoginAPI
from knox import views as knox_views
# from knox.views import LoginView as KnoxLoginView


urlpatterns = [
    url(r'^api/register/', RegisterAPI.as_view(), name='register'),
    url(r'^api/login/', LoginAPI.as_view(), name='login'),
    url(r'^api/logout/', knox_views.LogoutView.as_view(), name='logout'),
    url(r'^api/logoutall/', knox_views.LogoutAllView.as_view(), name='logoutall'),
]
