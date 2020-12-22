from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .models import previousProjects
from django.views.decorators.csrf import csrf_exempt
# from .serializers import projectapiSerializer
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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from Comparison import views as comparisonView
from bson import ObjectId


def readCsv(datasetName):
    print("Dataset Name", datasetName)
    if datasetName == "ISBSG":
        data = pd.read_csv(os.getcwd() +
                           '\\csv\\fully_final_1.csv')
    elif datasetName.__contains__("Promise"):
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
    # if(data.columns[-1].dtype == 'obj'):
    #     return data.columns[-1]

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
    print(a)
    new = pd.get_dummies(a)
    print(new)
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
    return Response({"columns": X.columns})


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgo(request):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    print("Request", request.data)
    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    datasetFile = request.data["csvFile"]
    classification = request.data['classificationType']
    data, X, y = readCsv(datasetFile)
    if(classification == "Binary"):
        y = conversion_to_defects(data)
    elif (classification == "Ternary"):
        y = convertToTernaryClassification(data)
    elif (classification == "Penta"):
        y = convertToPentaClassification(data)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []
    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=40)
    model = mlAlgoList(mlAlgo)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    result = model.predict([[float(i) for i in featuresValues]])
    # For Post Prediction
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    if(classification == 'Binary'):
        lr_auc = roc_auc_score(y_test, lr_probs)
    else:
        lr_auc = multiclass_roc_auc_score(y_test, lr_probs)
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    roc = conversion(lr_fpr, lr_tpr)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, prediction), auc(lr_recall, lr_precision)
    auc = conversion(lr_recall, lr_precision)
    # End Post Prediction
    if(classification == "Binary"):
        response = resultOfMl(result, auc, roc, y_test, prediction)
        return Response(response)
    elif (classification == "Ternary"):
        response = resultOfMl(result, auc, roc, y_test, prediction)
        return Response(response)
    elif (classification == "Penta"):
        response = resultOfMl(result, auc, roc, y_test, prediction)
        return Response(response)


def resultOfMl(result, auc, roc, y_test, prediction):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    score = accuracy_score(y_test, prediction)
    f1_score = f1_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction, average='weighted')
    matrix = confusion_matrix(y_test, prediction)
    report = classification_report(
        y_test, prediction, output_dict=True)
    if(result[0] == 0):
        res = "No Defects Detected"
    elif(result[0] == 1):
        res = "Defects Detected"
    a = {
        "result": res,
        "score": int(score*100),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matrix": matrix,
        "report": report,
        "roc": roc,
        "auc": auc
    }
    return (a)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    from sklearn import preprocessing
    # from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    # y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def applyMLAlgoWithRegression(request):
    print("applyMLAlgoWithRegression", request.data)
    features = request.data['features']
    mlAlgo = request.data['mlAlgo']
    datasetFile = request.data["csvFile"]
    data, X, y = readCsv(datasetFile)
    sortedArray = sorted(features.items())
    featuresNames = []
    featuresValues = []
    for i in sortedArray:
        featuresNames.append(i[0])
        featuresValues.append(i[1])
    X = data[featuresNames]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = mlAlgoListForRegression(mlAlgo)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    result = model.predict([[float(i) for i in featuresValues]])
    score = model.score(X_test, y_test)
    print("Score: ", score)
    result = model.predict([[float(i) for i in featuresValues]])
    print("Result", result[0])
    a = {"result": int(result),
         "score": int(score*100)
         }
    return Response(a)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def projectapi_getFeatures(request):
    method = request.data["method"]
    # datasetName = request.data["csvFile"]
    print(request.data)
    list = comparisonView.returnFeatuesList(
        method, 'decisiontree', request.data['csvFile'])
    print(list)
    return Response(list)


def projectapi_getFeatures1(method):
    method = request.data["method"]
    print(request.data)
    list = comparisonView.returnFeatuesList(method, 'decisiontree')
    print(list)
    return (list)


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


def conversion(arr1, arr2):
    arr = []
    print(arr2)
    for z in range(len(arr2)):
        arr.append({"x": arr1[z], "y": arr2[z]})
    return arr


def mlAlgoListForRegression(mlAlgo):
    if(mlAlgo == 'Decision Tree'):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
    elif(mlAlgo == 'Logestic Regression'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif(mlAlgo == 'K-Nearest Neighbors(KNN)'):
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
    elif (mlAlgo == 'Extra Trees'):
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=300)
    elif (mlAlgo == 'Random Forest'):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300)
    elif (mlAlgo == 'Ada Boost'):
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=500)
    return model


def mlAlgoList(mlAlgo):
    if(mlAlgo == 'Decision Tree'):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    elif(mlAlgo == 'Logestic Regression'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif(mlAlgo == 'K-Nearest Neighbors(KNN)'):
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
        model = SVC(probability=True)
    elif (mlAlgo == 'Linear Regression'):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif (mlAlgo == 'Extra Trees'):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=300, random_state=2)
    elif (mlAlgo == 'Random Forest'):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    elif (mlAlgo == 'Ada Boost'):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=500)
    return model


@csrf_exempt
@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def savePreviousProjects(request):
    print(request.data)
    # _id = request.data["_id"]
    user_id = request.data['user_id']
    state = request.data['state']
    log = previousProjects(user_id=user_id, state=state)
    log.save()
    return Response("Inserted")


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getUserlog(request):
    print(request.data)
    # user_id = request.data['id']
    # logs = previousProjects.objects.all().filter(user_id=userName)

    userName = request.data['username']

    logs1 = previousProjects.objects.all()
    print("data: ", logs1[0]._id)

    logs = previousProjects.objects.all().filter(user_id=userName)
    data = [{'userName': record.user_id, "log_id": str(record._id), 'state': record.state}
            for record in logs]
    return Response(data)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def deleteUserlog(request):
    print(request.data)
    user_id = request.data['id']
    log = previousProjects.objects.get(_id=ObjectId(user_id))
    print("Logs", log.user_id)
    log.delete()
    return Response("Successfully Deleted")
