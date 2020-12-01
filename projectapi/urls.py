# from django.urls import path
# from .views import registration
from .views import projectapi_getFeatures, applyMLAlgo, getFeaturesNames, applyBaggingAlgo, applyBoostingAlgo, applyMLAlgoWithRegression
from django.conf.urls import url, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    url(r'^getFeatures/', projectapi_getFeatures, name='Features'),
    url(r'^apply/', applyMLAlgo, name='MLAlgo'),
    url(r'^getFeaturesForMLComparison/', getFeaturesNames, name='getFeatures'),
    url(r'^applyRegression/', applyMLAlgoWithRegression, name='Regression'),
    url(r'^applyBagging/', applyBaggingAlgo, name='applyBaggingAlgo'),
    url(r'^applyBoosting/', applyBoostingAlgo, name='applyBoostingAlgo'),



    # url(r"^token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    # url(r"^refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # url(r'^login/', login, name="login"),
]
