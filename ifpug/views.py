from django.http import JsonResponse
from rest_framework.decorators import api_view
# Create your views here.


@api_view(['POST', ])
def ifpug(request):
    total = 0
    data = request.data['data']
    for d in data:
        total += d['score']
    newData = {"data": data, "total": total}
    return JsonResponse(newData, safe=False)
