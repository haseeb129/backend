from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
# from .models import projectapi
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from rest_framework import response, decorators, permissions, status


def readCsv():
    data = pd.read_csv(
        './csv/' + "Final_numaric_BinaryClassification.csv")
    X = data.drop(["Defect Density"], axis=1)
    y = data["Defect Density"]
    # X = X.drop(["Total Defects Delivered"], axis=1)
    return data, X, y


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def areaUnderCurve(request):

    # precision-recall curve and f1 for an imbalanced dataset
    # generate 2 class dataset
    # X, y =  make_classification(n_samples=1000, n_classes=2, weights=[0.99,0.01], random_state=1)
    # split into train/test sets
    data, X, y = readCsv()
    trainX, testX, trainy, testy = train_test_split(
        X, y, test_size=0.5, random_state=2)
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(testX)
    # calculate precision and recall for each threshold
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    # calculate scores
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
