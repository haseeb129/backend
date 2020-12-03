from .views import dmFeatureComparison, getTwoFeaturesNames, getTwoMLAlgoNames, getFeaturesNames, inputValueComparisonML, comparisonOfAllMLAlgo, withInputValuesComparisonML
from django.conf.urls import url, include

urlpatterns = [
    url(r'^comparisonAll/', comparisonOfAllMLAlgo, name='dmFeatureComparison'),
    url(r'^inputValueComparisonAll/',
        withInputValuesComparisonML, name='dmFeatureComparison'),
    url(r'^getTwoFeaturesNames/', getTwoFeaturesNames, name='dmFeatureComparison'),
    url(r'^getTwoMLAlgoNames/', getTwoMLAlgoNames, name='mlComparison'),
    url(r'^getFeaturesForMLComparison/', getFeaturesNames, name='getFeatures'),
    url(r'^inputValueComparisonML/', inputValueComparisonML, name='getFeatures'),
]
