from django.http import JsonResponse
from .models import cocomoBasic, Intermediatecocomo, detailedcocomo
from rest_framework.decorators import api_view

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm

from django.contrib.auth import authenticate, login, logout

from django.contrib import messages


# Create your views here.
# from .models import *
from .forms import CreateUserForm


from django.shortcuts import render
from rest_framework.generics import GenericAPIView
from .serializers import UserSerializer, LoginSerializer
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.contrib import auth
import jwt

from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
# from jwt.models import AuthToken

# @csrf_exempt


class RegisterView(GenericAPIView):
    serializer_class = UserSerializer

    # @ensure_csrf_cookie
    # method_decorator(csrf_protect)
    # def get(self, request):
    #     print(request)
    @csrf_exempt
    def post(self, request):
        # { % csrf_token % }
        # def ___init__(self, name):
        #     self.name = name
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            # jwt.
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(GenericAPIView):
    serializer_class = LoginSerializer

    def post(self, request):
        data = request.data
        username = data.get('username', '')
        password = data.get('password', '')
        user = auth.authenticate(username=username, password=password)

        if user:
            auth_token = jwt.sign(
                {'username': user.username}, "secret", {expiresIn: "5h"})

            serializer = UserSerializer(user)

            data = {'user': serializer.data, 'token': auth_token}

            return Response(data, status=status.HTTP_200_OK)

            # SEND RES
        return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


# def registerPage(request):
#     if request.user.is_authenticated:
#         return redirect('home')
#     else:
#         form = CreateUserForm()
#         if request.method == 'POST':
#             form = CreateUserForm(request.POST)
#             if form.is_valid():
#                 form.save()
#                 user = form.cleaned_data.get('username')
#                 messages.success(request, 'Account was created for ' + user)

#                 return redirect('login')

#         context = {'form': form}
#         return render(request, 'accounts/register.html', context)


# def loginPage(request):
#     if request.user.is_authenticated:
#         return redirect('home')
#     else:
#         if request.method == 'POST':
#             username = request.POST.get('username')
#             password = request.POST.get('password')

#             user = authenticate(request, username=username, password=password)

#             if user is not None:
#                 login(request, user)
#                 return redirect('home')
#             else:
#                 messages.info(request, 'Username OR password is incorrect')

#         context = {}
#         return render(request, 'accounts/login.html', context)


# def logoutUser(request):
#     logout(request)
#     return redirect('login')


# def home(request):
#     # orders = Order.objects.all()
#     # customers = Customer.objects.all()

#     # total_customers = customers.count()

#     # total_orders = orders.count()
#     # delivered = orders.filter(status='Delivered').count()
#     # pending = orders.filter(status='Pending').count()

#     # context = {'orders':orders, 'customers':customers,
#     # 'total_orders':total_orders,'delivered':delivered,
#     # 'pending':pending }
#     context = {}
#     return render(request, 'accounts/dashboard.html', context)


@api_view(['POST', ])
def basicCOCOMO(request):
    kloc = int(request.data['kloc'])
    data = cocomoBasic(kloc)
    retdata = {
        'effort': data[0],
        'devlopmentTime': data[1],
        'staffSize': data[2],
        'productivity': data[3]
    }
    return JsonResponse(retdata)


@api_view(['POST', ])
def IntermediateCOCOMO(request):
    kloc = int(request.data['kloc'])
    EAF = int(request.data['EAF'])
    data = Intermediatecocomo(kloc, EAF)
    retdata = {
        'effort': data[0],
        'devlopmentTime': data[1],
        'staffSize': data[2],
        'productivity': data[3]
    }
    return JsonResponse(retdata)


@api_view(['POST', ])
def detailedCOCOMO(request):
    kloc = int(request.data['kloc'])
    EAF = int(request.data['EAF'])
    U = int(request.data['U'])
    T = int(request.data['T'])
    data = detailedcocomo(kloc, EAF, U, T)
    retdata = {
        'effort': data[0],
        'totalEffort': data[1],
        'devlopmentTime': data[2],
        'totalDevolopmentTime': data[3],
        'staffSize': data[4],
        'productivity': data[5]
    }
    return JsonResponse(retdata)
