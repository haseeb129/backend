from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
# import pandas as pd

import os
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
from projectapi import views as projectapiView


def readCsv():
    # datasetName == "ISBSG"
    data = pd.read_csv(os.getcwd() +
                       '\\csv\\fully_final_1.csv')
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]
    return data, X, y


def conversion_to_defects(data):
    def checkDefects(col):
        if float(col) != 0:
            if float(col) <= 10:
                return "low"
            elif float(col) > 10 and float(col) <= 15:
                return "mediam"
            else:
                return "high"
            pass
        else:
            return 'Defects Present'

    def invertValues(col):
        if float(col) == 0:
            return 1
        else:
            return 0
    a = data["Defect Density"].apply(checkDefects)
    new = pd.get_dummies(a)
    # print(new)
    new["Defects Present"] = new["Defects Present"].apply(invertValues)
    defect_present = new['Defects Present']
    y = defect_present
    return y


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getFeaturesNames(request):
    print("Request getFeaturesNames", request.data)
    dataset = request.data['datasetName']
    data, X, y = readCsv()
    return Response(data.columns)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def comparisonOfAllMLAlgo(request):
    print(request.data)
    listOfMlAlgo = request.data["list"]
    features = request.data['features']
    list = []
    for mlAlgo in listOfMlAlgo:
        print(mlAlgo)
        accuracy_score = applyMLAlgoWithoutInputValues(mlAlgo, features)
        list.append({"MLName": mlAlgo, "score": accuracy_score})
    return Response(list)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def withInputValuesComparisonML(request):
    print(request.data)
    listOfMlAlgo = request.data["list"]
    features = request.data['inputFields']
    list = []
    for mlAlgo in listOfMlAlgo:
        accuracy_score, result = applyMLAlgo(mlAlgo, features)
        list.append(
            {"MLName": mlAlgo, "score": accuracy_score, "result": int(result[0])})
    return Response(list)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getTwoMLAlgoNames(request):
    print(request.data)
    print("PAth.      :", os.getcwd())
    ml1 = applyMLAlgoWithoutInputValues(
        request.data["m1"],
        request.data["features"],
    )
    ml2 = applyMLAlgoWithoutInputValues(
        request.data["m2"],
        request.data["features"],
        # request.data["targetClass"],
    )
    a = {"ML1": ml1, "ML2": ml2}
    return Response(a)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def inputValueComparisonML(request):
    print("inputValueComparisonML", request.data)

    print("PAth.      :", os.getcwd())
    ml1 = applyMLAlgo(

        request.data["MLAlgorithm1"],
        request.data["inputFields"],
    )
    ml2 = applyMLAlgo(

        request.data["MLAlgorithm2"],
        request.data["inputFields"],
    )
    a = {"ML1": ml1, "ML2": ml2}
    return Response(a)


def applyMLAlgoWithoutInputValues(mlAlgo, features):
    data, X, y1 = readCsv()
    y = conversion_to_defects(data)
    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = projectapiView.mlAlgoList(mlAlgo)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    # print("After 1")
    score = accuracy_score(y_test, prediction.round())
    print(score)
    return score


def applyMLAlgo(mlAlgo, features):
    data, X, y = readCsv()
    y = conversion_to_defects(data)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []

    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = projectapiView.mlAlgoList(mlAlgo)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction.round())
    print("Score : ", score)
    result = model.predict([[float(i) for i in featuresValues]])
    print("Result : ", result)
    return score, result


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getTwoFeaturesNames(request):
    print(request.data)
    features1 = dmFeatureComparison(
        request.data["a1"],
        request.data["a2"],
        request.data['csvFile']
    )
    features2 = dmFeatureComparison(
        request.data["b1"],
        request.data["b2"],
        request.data['csvFile']
    )
    a = {"First": features1, "Second": features2}
    return Response(a)


def dmFeatureComparison(method1, method2, datasetName):

    a = returnFeatuesList(
        method1,
        method2,
        datasetName,
    )
    return a


def returnFeatuesList(method, method1, datasetName):
    data, X, y = projectapiView.readCsv(datasetName)
    if(datasetName == 'isbsg'):
        encoded = conversion_to_defects(data)
    else:
        encoded = y
    if(method == 'Filter Method (Kbest)'):
        featureList = kbest(X, encoded)
        return featureList
    elif method == 'Wrapper Method (Recursive)':
        if(method1 == 'svr'):
            featureList = recursive(method1,  X, y)
            return featureList
        elif method1 == 'decisiontree':
            featureList = recursive(method1, X, encoded)
            return featureList
        elif method1 == 'logesticregression':
            featureList = recursive(method1, X, encoded)
            return featureList
        else:
            pass
    elif method == 'Embedded Method (Ridge)':
        featureList = embedded(X, y)
        return featureList
    else:
        pass


def recursive(method, X, y):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeClassifier
    print(method)
    if(method == 'svr'):
        model = SVR(kernel="linear")
    elif method == 'decisiontree':
        model = DecisionTreeClassifier()
    elif method == 'logesticregression':
        model = LogisticRegression()
    rfe = RFE(model, 10, step=1)
    fitRecursive = rfe.fit(X.abs(), y)
    d2 = {'Feature': X.columns, "Score": fitRecursive.ranking_}
    df2 = pd.DataFrame(d2)
    df2['Score'] = df2['Score']
    a = df2
    d = dict(a[a["Score"] == 1])
    return d["Feature"]


def kbest(X, y):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import chi2
    test = SelectKBest(chi2, k=10)
    fit = test.fit(X.abs(), y)
    d = {'Feature': X.columns.values, "Score": fit.scores_}
    df = pd.DataFrame(d)
    df['Score'] = df['Score'].astype(int)
    import heapq
    import random
    a = dict(df.nlargest(10, ['Score']))
    return a["Feature"]


def embedded(X, y):
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    d = {'Feature': X.columns, "Score": ridge.coef_}
    df = pd.DataFrame(d)
    # df
    df['Score'] = df['Score'].astype(int)
    a = df.sort_values(by='Score', ascending=False)
    d = dict(a[a["Score"] == 0])
    return d["Feature"]
