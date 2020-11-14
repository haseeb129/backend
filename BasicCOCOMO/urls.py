from django.urls import path
from . import views
from .views import RegisterView, LoginView

urlpatterns = [
    path('',  views.basicCOCOMO, name='basicCOCOMO'),
    path('register', RegisterView.as_view()),
    path('login', LoginView.as_view()),

    # path('register/', views.registerPage, name="register"),
    # path('login/', views.loginPage, name="login"),
    # path('logout/', views.logoutUser, name="logout"),
]
