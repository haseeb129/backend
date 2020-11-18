"""DefectPrediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path, include
from BasicCOCOMO import views
from django.conf.urls import url, include as i
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('BasicCOCOMO/', include('BasicCOCOMO.urls')),
    path('IntermediateCOCOMO/', include('BasicCOCOMO.IntermediateCOCOMO')),
    path('detailedCOCOMO/', include('BasicCOCOMO.detailedCOCOMO')),
    path('COCOMO2/', include("COCOMO2.urls")),
    path('ifpug/', include("ifpug.urls")),
    # path('register/', views.RegisterView, name="register"),
    # path('login/', views.LoginView, name="login"),
    path('api-auth/', include("rest_framework.urls")),
    path('api/token/', TokenObtainPairView.as_view()),
    path('api/token/refresh', TokenRefreshView.as_view()),
    path('api/jwtauth/', include('jwtauth.urls'), name='jwtauth'),
    path('api/defect_prediction/',
         include("projectapi.urls"), name="Defect_Prediction"),
    path('uploadfile/', include("UploadFile.urls")),
]
