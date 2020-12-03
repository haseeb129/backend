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


def readCsv():
    # datasetName == "ISBSG"
    data = pd.read_csv(os.getcwd() +
                       '\\csv\\fully_final_1.csv')
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]

    # data = pd.read_csv(
    #     os.getcwd()+'\\Comparison\\final_numeric_without_null.csv', decimal=',')
    # X = data.drop(["Defect Density"], axis=1)
    # y = data["Defect Density"]
    # X = X.drop(["Total Defects Delivered"], axis=1)
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
        accuracy_score = applyMLAlgo(mlAlgo, features)
        list.append({"MLName": mlAlgo, "score": accuracy_score})
    return Response(list)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getTwoMLAlgoNames(request):
    print(request.data)
    print("PAth.      :", os.getcwd())
    ml1 = applyMLAlgoWithoutInputValues(
        request.data["m1"],
        request.data["features"],
        # request.data["targetClass"],
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
    # features = data[features]
    # sortedArray = sorted(features.items())
    # featuresNames = []
    # featuresValues = []

    # for i in sortedArray:
    #     featuresNames.append(i[0])
    #     featuresValues.append(i[1])
    X = data[features]
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
    score = accuracy_score(y_test, prediction.round())
    print("Score : ", score)
    result = model.predict([[float(i) for i in featuresValues]])
    print("Result : ", result)

    a = {"result": result[0],
         "Accuracy_Score": score,

         }
    return a


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getTwoFeaturesNames(request):
    # method = request.data["method"]
    print(request.data)
    features1 = dmFeatureComparison(
        request.data["a1"],
        request.data["a2"],
        request.data['featuresCount'],
        # request.data['targetClass'],
    )
    features2 = dmFeatureComparison(
        request.data["b1"],
        request.data["b2"],
        request.data['featuresCount'],
        # request.data['targetClass'],
    )
    a = {"First": features1, "Second": features2}
    return Response(a)
    # return Response {"First": features1, "Second": features2}

# @decorators.api_view(["POST"])
# @decorators.permission_classes([permissions.AllowAny])


def dmFeatureComparison(method1, method2, featuresCount):
    # method = request.data["method"]
    # print(request.data)
    # print("X ::", X, y)

    a = returnFeatuesList(
        featuresCount,
        method1,
        method2,
        # target
        # X, y
        # request.data['X'],
        # request.data['y'],
    )
    # print(a)
    return a
    # return Response(a)


def returnFeatuesList(numberOfFeatures, method, method1):
    # print(numberOfFeatures, method, method1, X, y)
    data, X, y = readCsv()
    # y = data[target]
    encoded = conversion_to_defects(data)
    if(method == 'filter'):
        featureList = kbest(numberOfFeatures, X, encoded)
        return featureList
    elif method == 'wrapper':
        if(method1 == 'svr'):
            featureList = recursive(method1, numberOfFeatures, X, y)
            return featureList
        elif method1 == 'decisiontree':
            featureList = recursive(method1, numberOfFeatures, X, y)
            return featureList
        elif method1 == 'logesticregression':
            featureList = recursive(method1, numberOfFeatures, X, y)
            return featureList
        else:
            pass
    elif method == 'Embedded Method (Ridge)':
        featureList = embedded(numberOfFeatures, X, y)
        return featureList
    else:
        pass


def recursive(method, count, X, y):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeClassifier
    if(method == 'svr'):
        model = SVR(kernel="linear")
    elif method == 'decisiontree':
        model = DecisionTreeClassifier()
    elif method == 'logesticregression':
        model = LogisticRegression()
    rfe = RFE(model, count)
    fitRecursive = rfe.fit(X, y)
    d2 = {'Feature': X.columns, "Score": fitRecursive.ranking_}
    df2 = pd.DataFrame(d2)
    df2['Score'] = df2['Score']
    a = df2
#     print(a)
    d = dict(a[a["Score"] == 1])
#     print(d)
    return d["Feature"]


def kbest(count, X, y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    test = SelectKBest(score_func=chi2, k=count)
    # X1 = pd.DataFrame()
    for i in X.columns:
        # X[i] = pd.to_numeric(X[i], errors='coerce').fillna(0, downcast='infer')
        # X[i] = X[i].values.reshape(2, 2)
        X[i] = pd.to_numeric(X[i], errors='coerce')
    print(X.info())
    # X=X.reshape(-1,-1)
    fit = test.fit(X.abs(), y)
    d = {'Feature': X.columns.values, "Score": fit.scores_}
    df = pd.DataFrame(d)
    df['Score'] = df['Score'].astype(int)
    import heapq
    import random
    a = dict(df.nlargest(10, ['Score']))
    return a["Feature"]

# def kbest(count, X, y):
#     from sklearn.feature_selection import SelectKBest
#     from sklearn.feature_selection import chi2
#     test = SelectKBest(score_func=chi2, k=count)
#     # print(count, X, y)
#     # X1 = pd.DataFrame()
#     # for i in X:
#     #     X1.append(pd.DataFrame(columns=pd.to_numeric(i, errors='coerce')))
#     # a = X.abs()
#     # X1 = X.to_numeric(errors='coerce')
#     data, X, y1 = readCsv()
#     # print("rkhgkrhfjk", a)
#     # try:
#     # print(X.info())
#     for i in X.columns:
#         X[i] = pd.to_numeric(X[i], errors='coerce')
#     print(X.info())
#     fit = test.fit(X.abs(), y)
#     # except:
#     print("After fit")
#     d = {'Feature': X.columns.values, "Score": fit.scores_}
#     df = pd.DataFrame(d)
#     df['Score'] = df['Score'].astype(int)
#     # import heapq
#     # import random
#     a = dict(df.nlargest(10, ['Score']))
#     return a["Feature"]


def embedded(count, X, y):
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
