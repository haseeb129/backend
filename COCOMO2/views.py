from django.http import JsonResponse
from rest_framework.decorators import api_view
from .models import cocomo2
# Create your views here.


@api_view(['POST', ])
def COCOMO2(request):
    ob = int(request.data['ob'])
    vb = int(request.data['vb'])
    prod = int(request.data['prod'])
    data = cocomo2(ob, vb, prod)
    retdata = {
        'nop': data[0],
        'Effort': data[1]
    }
    return JsonResponse(retdata)
