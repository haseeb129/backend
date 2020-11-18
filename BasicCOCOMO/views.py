from django.http import JsonResponse
from rest_framework.response import Response
from .models import cocomoBasic, Intermediatecocomo, detailedcocomo
from rest_framework.decorators import api_view
from rest_framework import decorators, permissions


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def basicCOCOMO(request):
    kloc = int(request.data['kloc'])
    data = cocomoBasic(kloc)
    return Response(data)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def IntermediateCOCOMO(request):
    kloc = int(request.data['kloc'])
    EAF = float(request.data['EAF'])
    data = Intermediatecocomo(kloc, EAF)
    return Response(data)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def detailedCOCOMO(request):
    kloc = int(request.data['kloc'])
    EAF = float(request.data['EAF'])
    Name = request.data["name"]
    data = detailedcocomo(kloc, EAF, Name)
    retdata = {
        'effort': data[0],
        'devlopmentTime': data[1],

    }
    return JsonResponse(retdata)
