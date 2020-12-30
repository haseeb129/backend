import os as os
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.parsers import FileUploadParser
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from rest_framework import decorators, permissions
import pandas as pd
import numpy as np
from projectapi import views as projectApiView


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def upload(request):
    # './csv/'
    print("request", request.data)
    datasetName = request.data['datasetName']
    data, X, y = projectApiView.readCsv(datasetName)
    columns = data.columns
    return Response({"pandafile": data, "columns": columns, "info": data.info(verbose=False), "filePath": datasetName})


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def datasetStats(request):
    print("request", request.data)
    datasetName = request.data['datasetName']
    data, X, y = projectApiView.readCsv(datasetName)
    colNames, colType, colNullValues, colNotNullValues = datasetDetails(data)
    return Response({"colNames": colNames, "colType": colType, "colNullValues": colNullValues, "colNotNullValues": colNotNullValues})


def datasetDetails(data):
    import json
    # import self
    colNames = []
    colType = []
    colNullValues = []
    colNotNullValues = []
    colInstanses = []
    countNull = 0
    for col in data:
        colNames.append(col)
        for i in data[col]:
            # print("in for")
            if(i == "Null" or i == 123456789):
                # print("Null Values", countNull)
                countNull += 1
                # print(countNull)

        if(str(data[col]).__contains__("float")):
            colType.append("float")
        if(str(data[col]).__contains__("object")):
            colType.append("object")
        if(str(data[col]).__contains__("int")):
            colType.append("int")
        if(str(data[col]).__contains__("bool")):
            colType.append("boolean")
        colNotNullValues.append(data[col].value_counts().sum()-countNull)
        colNullValues.append(countNull)
        countNull = 0
    return colNames, colType, colNullValues, colNotNullValues


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def columnDetails(request):
    print("columnDetails Request", request.data)
    datasetName = request.data['datasetName']
    data, X, y = projectApiView.readCsv(datasetName)
    # pandafile = pd.read_csv('./csv/'+filePath)
    column = request.data['column']
    dataColumn = data[column]
    dic = dataColumnDetails(dataColumn)
    # if(dataColumn.dtypes == object):
    #     # print("Dealing with object")
    #     # nullvales = (dataColumn == '').sum()
    #     dic =objectColumnsDetails(dataColumn)

    #     # {"Null Values": nullvales, "Not Null Values": dataColumn.notnull().sum(
    #     #     axis=0)-nullvales, "type": "String/Object", "Unique Values": objetToDict(dataColumn.value_counts(dropna=False))}
    # else:
    #     dic =numaricColumnsDetails(dataColumn)
    # print("Dealing with Number")
    # print("column,", dic)

    return Response(dic)


def dataColumnDetails(dataColumn):
    if(dataColumn.dtypes == object):
        nullvales = (dataColumn == 'Null').sum()
        return {"Null Values": nullvales, "Not Null Values": dataColumn.notnull().sum(axis=0)-nullvales, "type": "String/Object", "Unique Values": objetToDict(dataColumn.value_counts(dropna=False))}
    else:
        nullvales = (dataColumn == 123456789).sum()
        return {"Min": dataColumn.min(),
                "dataColumn":dataColumn,
                "Max": dataColumn.max(),
                "Mean": dataColumn.mean(axis=0),
                "Mediam": dataColumn.median(axis=0),
                "dataColumn": dataColumn,
                "Standerd Deviation": dataColumn.std(axis=0),
                "Null Values": nullvales,
                "Not Null Values": dataColumn.notnull().sum(axis=0)-nullvales, "type": "INT-FLOAT"}


def numaricColumnsDetails(dataColumn):
    nullvales = (dataColumn == 123456789).sum()
    return {"Min": dataColumn.min(),
            "Max": dataColumn.max(),
            "Mean": dataColumn.mean(axis=0),
            "Mediam": dataColumn.median(axis=0),
            "Standerd Deviation": dataColumn.std(axis=0),
            "Null Values": nullvales,
            "Not Null Values": dataColumn.notnull().sum(axis=0)-nullvales, "type": "INT-FLOAT"}


# @decorators.api_view(["POST"])
# @decorators.permission_classes([permissions.AllowAny])
# def deleteColumn(request):
#     print("deleteColumn Request", request.data)
#     filePath = request.data['filePath']
#     pandafile = pd.read_csv(filePath)
#     column = request.data['column']
#     dataColumn = pandafile[column]
#     pandafile = pandafile.drop([request.data['column']], axis='columns')
#     savePandaFile1(pandafile, filePath)
#     return Response({"pandafile": pandafile, "columns": pandafile.columns, })


# @decorators.api_view(["POST"])
# @decorators.permission_classes([permissions.AllowAny])
# def fillNullValuesOfColumn(request):
#     print("fillNullValuesOfColumn Request", request.data)
#     filePath = request.data['filePath']
#     pandafile = pd.read_csv(filePath)
#     pandafile = make(pandafile)
#     column = pandafile[request.data['column']]
#     method = request.data['method']
#     if (method == "MEAN"):
#         stringValue = column.mean(axis=0)
#     elif (method == "MEDIAN"):
#         stringValue = column.median(axis=0)
#     elif (method == "REPLACE"):
#         stringValue = request.data['replaceValue']
#     pandafile[request.data['column']].replace(
#         123456789, float(stringValue), inplace=True)
#     savePandaFile1(pandafile, filePath)
#     return Response({"pandafile": pandafile, "columns": pandafile.columns,"ColumnDetails":dataColumnDetails(column) })


# @decorators.api_view(["POST"])
# @decorators.permission_classes([permissions.AllowAny])
# def fillNullValuesOfColumnString(request):
#     print("fillNullValuesOfColumnString Request", request.data)
#     filePath = request.data['filePath']
#     pandafile = pd.read_csv(filePath)
#     pandafile = make(pandafile)
#     column = pandafile[request.data['column']]
#     method = request.data['method']
#     if (method == "REPLACE"):
#         dataColumn = pandafile[request.data['column']
#                                ].replace("Null", request.data['replaceValue'], inplace=True)
#     elif (method == "REPLACEVALUE"):

#         pandafile[request.data['column']].replace(
#             request.data['to_replace_value'], request.data['to_value'], inplace=True)
#     savePandaFile1(pandafile, filePath)
#     return Response({"pandafile": pandafile, "columns": pandafile.columns,"ColumnDetails":dataColumnDetails(column) })


def make(pandafile):
    for i in pandafile.columns:
        if(pandafile[i].dtypes == object):
            # print("Dealing with object")
            pandafile[i] = pandafile[i].fillna("Null")
        else:
            # print("Dealing with Number")
            pandafile[i] = pandafile[i].fillna(123456789)

    return pandafile


def objetToDict(obj):
    keys = obj.keys()
    value = obj.values
    array = []
    i = 0
    while i < len(keys):

        key = keys[i]
        if(str(key) == ""):
            key = "Null"
        value1 = value[i]
        array.append({"name": str(key), "value": value1})
        i += 1
    return array


# def savePandaFile(pandafile, fileName):
#     df = pd.DataFrame(pandafile)
#     db = csvName(name=fileName)
#     db.save()
#     df.to_csv(fileName, index=False)


def savePandaFile1(pandafile, fileName):
    df = pd.DataFrame(pandafile)
    df.to_csv(fileName, index=False)
