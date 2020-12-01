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
                           '\\csv\\fully_final_1.csv')
    elif datasetName.__contains__("promise"):
        csv = datasetName.split(" ")
        # print(type(csv[1]))
        # print(type(os.getcwdb()))
        # a = str(os.getcwdb())+str('\\csv\\'+csv[1]+'.csv')
        # a = a.replace("'", "")
        # a = a.replace("b", "")
        # print("Full : ", a)
        # data = pd.read_csv(a)
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]
    # X = X.drop(["Total Defects Delivered"], axis=1)
    return data, X, y


def convertToTernaryClassification(data):
    def checkDefects(col):
        if col != 0:
            if col <= 20:
                return "Low"
            else:
                return "High"
        else:
            return 'Zero'
    a = data[data.columns[-1]].apply(checkDefects)
    return a


def convertToPentaClassification(data):
    def checkDefects(col):
        if col != 0:
            if col <= 20:
                return "Low"
            elif col > 20 and col <= 60:
                return "Medium"
            elif col > 60 and col <= 90:
                return "High"
            else:
                return "Very High"
        else:
            return 'Zero'
    a = data[data.columns[-1]].apply(checkDefects)
    return a


def conversion_to_defects(data):
    def checkDefects(col):
        if col != 0:
            if col <= 10:
                return "Low"
            elif col > 10 and col <= 15:
                return "Medium"
            else:
                return "High"
        else:
            return 'Defects Present'

    def invertValues(col):
        if col == 0:
            return 1
        else:
            return 0
    a = data[data.columns[-1]].apply(checkDefects)
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
    return Response({"columns": X.columns})


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgo(request):
    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    datasetFile = request.data["csvFile"]
    classification = request.data['classificationType']
    data, X, y = readCsv(datasetFile)
    if(classification == "Binary"):
        y = conversion_to_defects(data)
    elif (classification == "Ternary"):
        y = convertToTernaryClassification(data)
        print(y)
    elif (classification == "Penta"):
        y = convertToPentaClassification(data)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []

    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]
    # print(y)
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=40)
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
    result = model.predict([[float(i) for i in featuresValues]])
    if(classification == "Binary"):
        score = accuracy_score(y_test, prediction.round())
        matrix = confusion_matrix(y_test, prediction.round())
        report = classification_report(
            y_test, prediction.round(), output_dict=True)
        if(result[0] == 0):
            res = "No Defects Detected"
        else:
            res = "Defects Detected"
        a = {
            "result": res,
            "score": score,
            "matrix": matrix,
            "report": report

        }
        return Response(a)
    elif (classification == "Ternary"):
        score = accuracy_score(y_test, prediction)
        matrix = confusion_matrix(y_test, prediction)
        report = classification_report(
            y_test, prediction, output_dict=True)
    elif (classification == "Penta"):
        score = accuracy_score(y_test, prediction)
        matrix = confusion_matrix(y_test, prediction)
        report = classification_report(
            y_test, prediction, output_dict=True)
    print("Score: ", score)
    # print("After 2")
    # print("After 3")
    print("Result", result[0])

    a = {"result": result,
         "score": score,
         "matrix": matrix,
         "report": report

         }
    return Response(a)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgoWithRegression(request):
    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    datasetFile = request.data["csvFile"]
    # classification = request.data['classificationType']
    data, X, y = readCsv(datasetFile)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []

    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]
    # print(y)
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    if(mlAlgo == 'Decision Tree Regression'):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
    elif(mlAlgo == 'Logestic Regression'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif(mlAlgo == 'K-Nearest Neighbors(KNN) Regression'):
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=300)
    elif(mlAlgo == 'Linear Discriminant Analysis'):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif(mlAlgo == 'Naive Bayes (Gaussian NB)'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif(mlAlgo == 'Support Vector Machine (SVR)'):
        from sklearn.svm import SVR
        model = SVR()
    elif (mlAlgo == 'Linear Regression'):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif (mlAlgo == 'Extra Trees Regression'):
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=300)
    elif (mlAlgo == 'Random Forest Regression'):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300)
    elif (mlAlgo == 'Ada Boost Regression'):
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=500)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("After 1")
    # print("Prediction : ",prediction)
    # print(y_test)
    result = model.predict([[float(i) for i in featuresValues]])
    score = model.score(X_test, y_test)
    print("Score: ", score)
    result = model.predict([[float(i) for i in featuresValues]])
    print("Result", result[0])

    a = {"result": result,
         "score": score
         }
    return Response(a)


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
        X = ['Max Team Size', 'Project Elapsed Time', 'Language_Type_4GL', 'Development_Platform_MR',
             'Value Adjustment Factor', 'Normalised Work Effort', 'Adjusted Function Points', 'Functional Size', 'Input count']

    elif (method == 'Embedded Method (Ridge)'):
        X = ['Normalised Level 1 PDR (ufp)', 'Normalised Work Effort', 'Summary Work Effort',
             'Normalised Work Effort Level 1', 'Effort Unphased', 'Adjusted Function Points', 'Functional Size', 'Added count', 'Input count']

    return Response(X)


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
