# from django.urls import path
# from .views import registration
from .views import projectapi_getFeatures, applyMLAlgo
from django.conf.urls import url, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    url(r'^getFeatures/', projectapi_getFeatures, name='Features'),
    url(r'^apply/', applyMLAlgo, name='MLAlgo'),
    # url(r"^token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    # url(r"^refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # url(r'^login/', login, name="login"),
]