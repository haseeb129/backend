from django.contrib.auth import get_user_model
from rest_framework import permissions
from rest_framework import response, decorators, permissions, status
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserCreateSerializer, LoginSerializer, UserSerializer
from rest_framework import serializers
from rest_framework.decorators import api_view
import jwt
from rest_framework.response import Response
from django.conf import settings
from django.contrib import auth
from rest_framework.generics import GenericAPIView
from .models import User
User = get_user_model()


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def registration(request):
    print("Request: ", request.data)
    serializer = UserCreateSerializer(data=request.data)
    print(serializer.is_valid())
    if serializer.is_valid():
        print("before")

        error = serializer.save()
        print("validation error", error)
        auth_token = jwt.encode(
            {'username': serializer.data["username"], "email": serializer.data["email"]}, settings.SECRET_KEY)
        # print(user)
        # serializer = UserSerializer(user)
        print(serializer.data)
        data = {'user': serializer.data, 'token': auth_token}
        print("data is ", data)
        return Response(data, status=status.HTTP_200_OK)

        print("error")

        # return Response(serializer.data, status=status.HTTP_201_CREATED)
    print("After Try error", serializer.errors)
    # if(serializer.errors['username']):
    #     return Response({"message": "This Username already exist."}, status=status.HTTP_400_BAD_REQUEST)
    # elif(serializer.errors['email']):
    #     return Response({"message": "This email already exist."}, status=status.HTTP_400_BAD_REQUEST)
    # else:

    return Response({"message": "System Error."}, status=status.HTTP_400_BAD_REQUEST)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def login(request):
    serializer = LoginSerializer(data=request.data)
    data = request.data
    # print("User Data "+data["email"])
    username = data.get('username', '')
    password = data.get('password', '')
    id = data.get('_id', '')
    print("ID: ", id)
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


# def getAllUser()
@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def getAllUser(request):
    users = User.objects.all()
    userList = []
    data = [{'username': user.username, "email": str(user.email)}
            for user in users]
    print("data: ", users[0].email)
    return Response(data)


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def deleteUser(request):
    print(request.data)
    username = request.data['username']
    user = User.objects.get(username=username)
    # print("User: ", user)
    user.delete()
    return Response("Successfully Deleted")


@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def resetPassword(request):
    print(request.data)
    serializer = LoginSerializer(data=request.data)
    data = request.data
    username = data.get('username', '')
    password = data.get('password', '')
    newPassword = data.get('password1', '')
    # id = data.get('_id', '')
    # print("ID: ", id)
    user = auth.authenticate(username=username, password=password)
    print("Before: ",user)
    if user:
        print("After auth", user)
        u1 = User.objects.get(username=username)
        u1.set_password(newPassword)
        u1.save()
        auth_token = jwt.encode(
            {'username': u1.username}, settings.SECRET_KEY)
        # print(user)
        serializer = UserSerializer(u1)

        data = {'user': serializer.data, 'token': auth_token}
        print("data is ", data)
        return Response(data, status=status.HTTP_200_OK)
    else:
        return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
