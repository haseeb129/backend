from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .models import projectapi
from .serializers import projectapiSerializer
import csv
import io
import json
import os
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
import numpy as np
import pandas as pd
import sklearn as sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from rest_framework import response, decorators, permissions, status

def readCsv():
    data = pd.read_csv(os.getcwd()+
        '\\projectapi\\final_numeric_without_null.csv')
    X = data.drop(["Defect Density"], axis=1)
    y = data["Defect Density"]
    X = X.drop(["Total Defects Delivered"], axis=1)
    return data, X


def conversion_to_defects(data):
    def checkDefects(col):
        if col != 0:
            if col <= 10:
                return "low"
            elif col > 10 and col <= 15:
                return "mediam"
            else:
                return "high"
        else:
            return 'Defects Present'

    def invertValues(col):
        if col == 0:
            return 1
        else:
            return 0
    a = data["Defect Density"].apply(checkDefects)
    new = pd.get_dummies(a)
    new["Defects Present"] = new["Defects Present"].apply(invertValues)
    defect_present = new['Defects Present']
    y = defect_present
    return y



@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getFeaturesNames(request):
    print("Request getFeaturesNames",request.data)
    dataset=request.data['datasetName']
    data, X= readCsv()

    return Response(data.columns)



@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgo(request):

    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    data, X = readCsv()
    y = conversion_to_defects(data)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []

    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if(mlAlgo == 'decisiontree'):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif(mlAlgo == 'logesticregression'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    elif(mlAlgo == 'LinearDiscriminantAnalysis'):
        print("LinearDiscriminantAnalysis")
        model = LinearDiscriminantAnalysis()
    elif(mlAlgo == 'GaussianNB'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif(mlAlgo == 'svc'):
        from sklearn.svm import SVC
        model = SVC()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("After 1")
    score = accuracy_score(y_test, prediction)
    print(score)
    print("After 2")
    result = model.predict([featuresValues])
    print("After 3")
    print("Result", result)
    matrix = confusion_matrix(y_test, prediction)
    report = classification_report(y_test, prediction, output_dict=True)

    a = {"result": result,
         "score": score,
         "matrix": matrix,
         "report": report

         }
    return Response(a)

@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def projectapi_testing(request):
    method = request.data["method"]
    print(request.data)
    print(method)

    return Response(request.data["array"])


# @api_view(['POST', ])
@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def projectapi_getFeatures(request):
    method = request.data["method"]
    print(request.data)
    print(method)
    if(method == 'kbest'):
        X = ['Normalised Work Effort', 'Summary Work Effort', 'Normalised Work Effort Level 1', 'Effort Unphased',
             'Adjusted Function Points', 'Functional Size', 'Added count', 'Input count', 'Speed of Delivery', 'Normalised Level 1 PDR (ufp)']

    elif(method == 'recursive'):
        X = ['Year of Project', 'Max Team Size', 'Project Elapsed Time', 'Language_Type_4GL', 'Development_Platform_MR',
             'Value Adjustment Factor', 'Normalised Work Effort', 'Adjusted Function Points', 'Functional Size', 'Input count']

    elif (method == 'filter'):
        X = ['Normalised Level 1 PDR (ufp)', 'Year of Project', 'Normalised Work Effort', 'Summary Work Effort',
             'Normalised Work Effort Level 1', 'Effort Unphased', 'Adjusted Function Points', 'Functional Size', 'Added count', 'Input count']

    return Response(X)


@api_view(['GET', ])
def projectapi_view1(request):
    try:
        data = projectapi.objects.get()
    except projectapi.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    if request.method == "GET":
        serializer = projectapiSerializer(data)
        return Response(serializer.data)



@api_view(['POST', ])
def projectapi_view2(request):
    print("Normalized_Work_Effort",
          request.data["Normalized_Work_Effort"])
    try:
        data = projectapi.objects.get()
    except projectapi.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    if request.method == "POST":
        serializer = projectapiSerializer(data, request.data)
        if serializer.is_valid():
            serializer.save()
            print("Normalized_Work_Effort",
                  serializer.data["Normalized_Work_Effort"])
            Normalized_Work_Effort = serializer.data["Normalized_Work_Effort"]
            Summary_Work_Effort = serializer.data["Summary_Work_Effort"]
            Normalised_Work_Effort_Level_1 = serializer.data["Normalised_Work_Effort_Level_1"]
            Effort_Unphased = serializer.data["Effort_Unphased"]
            Adjusted_Function_Points = serializer.data["Adjusted_Function_Points"]
            Functional_Size = serializer.data["Functional_Size"]
            Added_count = serializer.data["Added_count"]
            Input_count = serializer.data["Input_count"]
            Max_Team_Size = serializer.data["Max_Team_Size"]
            Speed_of_Delivery = serializer.data["Speed_of_Delivery"]
            Development_Type_New_Development = serializer.data["Development_Type_New_Development"]
            Language_Type_3GL = serializer.data["Language_Type_3GL"]

            model = pd.read_pickle(
                r"C:\Users\hasee\Downloads\Logestic_Regression_Model.pickle")
            # Make prediction
            result = model.predict(
                [[Normalized_Work_Effort, Summary_Work_Effort, Normalised_Work_Effort_Level_1, Effort_Unphased, Adjusted_Function_Points, Functional_Size, Added_count, Input_count, Max_Team_Size, Speed_of_Delivery, Development_Type_New_Development, Language_Type_3GL]])

            classification = result[0]
            # classification = 0

            return Response(classification, status=status.HTTP_201_CREATED)
        return Response(status=status.HTTP_400_BAD_REQUEST)
