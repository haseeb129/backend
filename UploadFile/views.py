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
from .models import csvName


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def upload(request):
    print("request", request.data)
    parser_class = (FileUploadParser,)
    if 'file' not in request.data:  # if file not present
        raise ParseError("Empty content")
    f = request.data['file']
    filename = request.data['name']
    pandafile = pd.read_csv(f, encoding="ISO-8859-1", error_bad_lines=False)
    pandafile = make(pandafile)
    df = pd.DataFrame(pandafile)
    db = csvName(name=filename)
    path = './csv/'+filename
    columns=pandafile.columns
    df.to_csv(path, index=False, header=True)
    db.save()
    # df.to_csv(filename, index=False)
    # pand = pd.read_csv(path, encoding="ISO-8859-1",error_bad_lines=False)
    return Response({"pandafile":pandafile,"columns":columns,"info": pandafile.info(verbose=False),"filePath":path})


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def readUploadedFile(request):
    print("request", request.data)
    filePath = request.data['filePath']
    pandafile = pd.read_csv(filePath, encoding="ISO-8859-1", error_bad_lines=False)
    pandafile = make(pandafile)   
    columns=pandafile.columns
    return Response({"pandafile":pandafile,"columns":columns,"info": pandafile.info(verbose=False),"filePath":filePath})



@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def columnDetails(request):
    print("columnDetails Request", request.data)
    filePath = request.data['filePath']
    pandafile = pd.read_csv(filePath)
    column = request.data['column']
    dataColumn = pandafile[column]
    dic=dataColumnDetails(dataColumn)
    return Response(dic)


def dataColumnDetails(dataColumn):
    if(dataColumn.dtypes == object):
        nullvales = (dataColumn == 'Null').sum()
        return {"Null Values": nullvales, "Not Null Values": dataColumn.notnull().sum(axis=0)-nullvales, "type": "String/Object", "Unique Values": objetToDict(dataColumn.value_counts(dropna=False))}
    else:
        nullvales = (dataColumn== 123456789).sum()
        return {"Min": dataColumn.min(),
                    "Max": dataColumn.max(),
                    "Mean": dataColumn.mean(axis=0),
                    "Mediam": dataColumn.median(axis=0),
                    "Standerd Deviation": dataColumn.std(axis=0),
                    "Null Values": nullvales,
                    "Not Null Values": dataColumn.notnull().sum(axis=0)-nullvales, "type": "INT-FLOAT"}
 

@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def deleteColumn(request):
    print("deleteColumn Request", request.data)
    filePath = request.data['filePath']
    pandafile = pd.read_csv(filePath)
    column = request.data['column']
    dataColumn = pandafile[column]
    pandafile = pandafile.drop([request.data['column']], axis='columns')
    savePandaFile1(pandafile, filePath)
    return Response({"pandafile": pandafile, "columns": pandafile.columns, })


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def fillNullValuesOfColumn(request):
    print("fillNullValuesOfColumn Request", request.data)
    filePath = request.data['filePath']
    pandafile = pd.read_csv(filePath)
    pandafile = make(pandafile)
    column = pandafile[request.data['column']]
    method = request.data['method']
    if (method == "MEAN"):
        stringValue = column.mean(axis=0)
    elif (method == "MEDIAN"):
        stringValue = column.median(axis=0)
    elif (method == "REPLACE"):
        stringValue = request.data['replaceValue']
    pandafile[request.data['column']].replace(
        123456789, float(stringValue), inplace=True)
    savePandaFile1(pandafile, filePath)
    return Response({"pandafile": pandafile, "columns": pandafile.columns,"ColumnDetails":dataColumnDetails(column) })


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def fillNullValuesOfColumnString(request):
    print("fillNullValuesOfColumnString Request", request.data)
    filePath = request.data['filePath']
    pandafile = pd.read_csv(filePath)
    pandafile = make(pandafile)
    column = pandafile[request.data['column']]
    method = request.data['method']
    if (method == "REPLACE"):
        dataColumn = pandafile[request.data['column']
                               ].replace("Null", request.data['replaceValue'], inplace=True)
    elif (method == "REPLACEVALUE"):

        pandafile[request.data['column']].replace(
            request.data['to_replace_value'], request.data['to_value'], inplace=True)
    savePandaFile1(pandafile, filePath)
    return Response({"pandafile": pandafile, "columns": pandafile.columns,"ColumnDetails":dataColumnDetails(column) })


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


def savePandaFile(pandafile, fileName):
    df = pd.DataFrame(pandafile)
    db = csvName(name=fileName)
    db.save()
    df.to_csv(fileName, index=False)


def savePandaFile1(pandafile, fileName):
    df = pd.DataFrame(pandafile)
    df.to_csv(fileName, index=False)
