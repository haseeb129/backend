# jwtauth/views.py

from django.contrib.auth import get_user_model
from rest_framework import permissions
from rest_framework import response, decorators, permissions, status
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserCreateSerializer, LoginSerializer, UserSerializer

from rest_framework.decorators import api_view

import jwt
from rest_framework.response import Response


from django.conf import settings
from django.contrib import auth
from rest_framework.generics import GenericAPIView

User = get_user_model()


# class RegisterView(GenericAPIView):
#     # serializer_class = UserSerializer

#     def post(self, request):
#         serializer = UserCreateSerializer(data=request.data)
#         if not serializer.is_valid():
#             return response.Response(serializer.errors, status.HTTP_400_BAD_REQUEST)
#         user = serializer.save()
#         refresh = RefreshToken.for_user(user)
#         res = {
#             "refresh": str(refresh),
#             "access": str(refresh.access_token),
#         }
#         return response.Response(res, status.HTTP_201_CREATED)


# @decorators.api_view(["POST"])
# @decorators.permission_classes([permissions.AllowAny])
# def Do_Registration(request):
#     first_name = request.data["first_name"]
#     # print("Request = ", request.data)
#     last_name = request.data["last_name"]
#     user = User(first_name=first_name, last_name=last_name,
#                 email=request.data["email"], password=request.data["password"])
#     # post = Posts(post_title=request.data["post_title"],
#     #              post_description=request.data['post_description'], comment=comment, tags=tags, user_details=user_details)
#     # # post = Posts(user_details=user_details)
#     # post = Posts(tags=tags, comment=comment, user_details=user_details)
#     user.save()
#     return Response("Inserted")


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def registration(request):
    serializer = UserCreateSerializer(data=request.data)
    print(serializer.is_valid())
    if serializer.is_valid():
        print("before")
        serializer.save()
        print(serializer.data)
        auth_token = jwt.encode(
            {'username': serializer.data["username"], "email": serializer.data["email"]}, settings.SECRET_KEY)
        # print(user)
        # serializer = UserSerializer(user)

        data = {'user': serializer.data, 'token': auth_token}
        print("data is ", data)
        return Response(data, status=status.HTTP_200_OK)

        # return Response(serializer.data, status=status.HTTP_201_CREATED)
    print("error")
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # serializer = UserCreateSerializer(data=request.data)
    # print(serializer.is_valid())
    # if not serializer.is_valid():
    #     print("in if")
    #     return response.Response(serializer.errors, status.HTTP_400_BAD_REQUEST)
    # user = serializer.save()
    # print(user)
    # refresh = RefreshToken.for_user(user)
    # res = {
    #     "refresh": str(refresh),
    #     "access": str(refresh.access_token),
    # }
    # return response.Response(res, status.HTTP_201_CREATED)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def login(request):
    serializer = LoginSerializer(data=request.data)
    data = request.data
    # print("User Data "+data["email"])
    username = data.get('username', '')
    password = data.get('password', '')
    user = auth.authenticate(username=username, password=password)
    print("After auth", user)
    if user:
        auth_token = jwt.encode(
            {'username': user.username}, settings.SECRET_KEY)
        # print(user)
        serializer = UserSerializer(user)

        data = {'user': serializer.data, 'token': auth_token}
        print("data is ", data)
        return Response(data, status=status.HTTP_200_OK)

        # SEND RES
    return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

    # if not serializer.is_valid():
    #     return response.Response(serializer.errors, status.HTTP_400_BAD_REQUEST)
    # user = serializer.save()
    # print (user)
    # refresh = RefreshToken.for_user(user)
    # res = {
    #     "refresh": str(refresh),
    #     "access": str(refresh.access_token),
    # }
    # return response.Response(res, status.HTTP_201_CREATED)


# class LoginView(GenericAPIView):
#     serializer_class = LoginSerializer

#     def post(self, request):
#         data = request.data
#         username = data.get('username', '')
#         password = data.get('password', '')
#         user = auth.authenticate(username=username, password=password)

#         if user:
#             auth_token = jwt.encode(
#                 {'username': user.username}, settings.SECRET_KEY)
#             # print(user)
#             serializer = UserSerializer(user)

#             data = {'user': serializer.data, 'token': auth_token}

#             return Response(user, status=status.HTTP_200_OK)

#             # SEND RES
#         return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
