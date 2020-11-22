from . import views
from django.urls import path
urlpatterns = [
    path('', views.upload, name='upload'),
    
    path('readUploadedFile/', views.readUploadedFile, name='readUploadedFile'),
    path('columnDetails/', views.columnDetails, name='columnDetails'),
    path('deleteColumn/', views.deleteColumn, name='deleteColumn'),
    path('fillna/', views.fillNullValuesOfColumn, name='fillNullValuesOfColumn'),
    path('fillnastring/', views.fillNullValuesOfColumnString, name='fillnastring'),


]
