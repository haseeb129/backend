# from django.urls import path
# from .views import registration
from .views import registration, login, getAllUser, deleteUser,resetPassword
from django.conf.urls import url, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    # path('register/', registration, name='register'),
    url(r'^register/', registration, name='register'),
    url(r"^token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    url(r"^refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    url(r'^login/', login, name="login"),
    url(r'^getAllUser/', getAllUser, name="getAllUser"),
    url(r'^deleteUser/', deleteUser, name="deleteUser"),
    url(r'^resetPassword/', resetPassword, name="resetPassword")
]
