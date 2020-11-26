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
from sklearn.naive_bayes import GaussianNB
from rest_framework import response, decorators, permissions, status


def readCsv(datasetName):
    print("Dataset Name", datasetName)
    if datasetName == "ISBSG":
        data = pd.read_csv(os.getcwd() +
                           '\\csv\\Final_numaric_BinaryClassification.csv')
    elif datasetName.__contains__("promise"):
        csv = datasetName.split(" ")
        print(type(csv[1]))
        print(type(os.getcwdb()))
        a = str(os.getcwdb())+str('\\csv\\'+csv[1]+'.csv')
        a = a.replace("'", "")
        a = a.replace("b", "")
        print("Full : ", a)
        data = pd.read_csv(a)
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]
    # X = X.drop(["Total Defects Delivered"], axis=1)
    return data, X, y


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
    print("Request getFeaturesNames", request.data)
    dataset = request.data['csvFile']
    data, X, y = readCsv(dataset)

    # return Response(data.columns)
    return Response({"columns": X.columns, "correlation": X.corr()})


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgo(request):
    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    datasetFile = request.data["csvFile"]
    data, X, y = readCsv(datasetFile)
    # print(y)
    # y = conversion_to_defects(data)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []

    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]
    # print(y)
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if(mlAlgo == 'Decision Tree Classifier'):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif(mlAlgo == 'Logestic Regression'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif(mlAlgo == 'K-Nearest Neighbors(KNN) Classifier'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    elif(mlAlgo == 'Linear Discriminant Analysis'):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif(mlAlgo == 'Naive Bayes (Gaussian NB)'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif(mlAlgo == 'Support Vector Machine (SVM)'):
        from sklearn.svm import SVC
        model = SVC()
    elif (mlAlgo == 'Linear Regression'):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif (mlAlgo == 'Extra Trees Classifier'):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=300)
    elif (mlAlgo == 'Random Forest Classifier'):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300)
    elif (mlAlgo == 'Ada Boost Classifier'):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=500)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("After 1")
    # print("Prediction : ",prediction)
    # print(y_test)
    score = accuracy_score(y_test, prediction.round())
    print("Score: ", score)
    # print("After 2")
    result = model.predict([[float(i) for i in featuresValues]])
    # print("After 3")
    print("Result", result[0])
    matrix = confusion_matrix(y_test, prediction.round())
    report = classification_report(
        y_test, prediction.round(), output_dict=True)

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
    if(method == 'Filter Method (Kbest)'):
        X = ['Normalised Work Effort', 'Summary Work Effort', 'Normalised Work Effort Level 1', 'Effort Unphased',
             'Adjusted Function Points', 'Functional Size', 'Added count', 'Input count', 'Speed of Delivery', 'Normalised Level 1 PDR (ufp)']

    elif(method == 'Wrapper Method (Recursive)'):
        X = ['Year of Project', 'Max Team Size', 'Project Elapsed Time', 'Language_Type_4GL', 'Development_Platform_MR',
             'Value Adjustment Factor', 'Normalised Work Effort', 'Adjusted Function Points', 'Functional Size', 'Input count']

    elif (method == 'Embedded Method (Ridge)'):
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


def projectapi_getFeatures1(method):
    # method = request.data["method"]
    # print(request.data)
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

    return X


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyBaggingAlgo(request):
    model = request.data['model']
    numberOfEstimators = request.data['estimators']
    data, X = readCsv()
    method = "kbest"
    X = projectapi_getFeatures1(method)
    X = data[X]
    y = conversion_to_defects(data)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import BaggingClassifier
    from sklearn import model_selection
    if model == 'kneighbours':
        model1 = KNeighborsClassifier()
    elif model == 'decisiontree':
        model1 = DecisionTreeClassifier()
    elif model == 'svc':
        model1 = SVC()
    elif model == 'logestic':
        model1 = LogisticRegression()
    elif model == 'gaussian':
        model1 = GaussianNB()
    seed = 8
    kfold = model_selection.KFold(n_splits=3,
                                  random_state=int(numberOfEstimators))
    # initialize the base classifier
    base_cls = model1
    # no. of base classifier
    # num_trees = 500
    # bagging classifier
    model = BaggingClassifier(base_estimator=base_cls,
                              n_estimators=1000)
    results = model_selection.cross_val_score(model, X, y, cv=kfold)
    print("accuracy :")
    print(results.mean())
    return Response(results.mean())


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyVotingAlgo(request):

    model = VotingClassifier(
        estimators=[('lr', model1), ('dt', model2), ("svc", model3)], voting='hard')
    model.fit(X_train, y_train)
    model.score(X_test, y_test)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyBoostingAlgo(request):
    model = request.data['ensemblingMethod']
    numberOfEstimators = request.data['estimators']
    data, X = readCsv()
    method = "kbest"
    X = projectapi_getFeatures1(method)
    X = data[X]
    y = conversion_to_defects(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    if model == 'gradient':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=numberOfEstimators)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        a = clf.score(X_test, y_test)
        print(a)
    elif model == 'extratree':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=numberOfEstimators)
        model.fit(X_train, y_train)
        a = model.score(X_test, y_test)
    elif model == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=numberOfEstimators)
        model.fit(X_train, y_train)
        a = model.score(X_test, y_test)
    elif model == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=500)
        model.fit(X_train, y_train)
        a = model.score(X_test, y_test)
    return Response(a)
