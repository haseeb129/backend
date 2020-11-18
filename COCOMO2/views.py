from django.http import JsonResponse
from rest_framework.decorators import api_view
from .models import cocomo2
from rest_framework.response import Response
from rest_framework import decorators, permissions
# Create your views here.


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def COCOMO2(request):
    kloc = int(request.data['kloc'])
    scaleSum = float(request.data['scaleSum'])
    EAF = float(request.data['EAF'])
    B = 0.91 + 0.01 * scaleSum
    SE = 0.3179
    effort = 2.94*EAF*((kloc)**B)
    duration = 3.67*(effort**SE)
    staff = effort/duration
    return Response({"effort": effort, "duration": duration, "staff": staff})


#
# def COCOMO2(request):
#     ob = int(request.data['ob'])
#     vb = int(request.data['vb'])
#     prod = int(request.data['prod'])
#     data = cocomo2(ob, vb, prod)
#     retdata = {
#         'nop': data[0],
#         'Effort': data[1]
#     }
#     return JsonResponse(retdata)
