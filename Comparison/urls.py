from .views import dmFeatureComparison, getTwoFeaturesNames, getTwoMLAlgoNames, getFeaturesNames,inputValueComparisonML
from django.conf.urls import url, include

urlpatterns = [
    url(r'^getTwoFeaturesNames/', getTwoFeaturesNames, name='dmFeatureComparison'),
    url(r'^getTwoMLAlgoNames/', getTwoMLAlgoNames, name='mlComparison'),
    url(r'^getFeaturesForMLComparison/', getFeaturesNames, name='getFeatures'),
    url(r'^inputValueComparisonML/', inputValueComparisonML, name='getFeatures'),
]
