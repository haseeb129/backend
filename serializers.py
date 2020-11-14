# jwtauth/serializers.py

from django.contrib.auth import get_user_model
from rest_framework import serializers

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        max_length=65, min_length=8, write_only=True)
    email = serializers.EmailField(max_length=255, min_length=4),
    first_name = serializers.CharField(max_length=255, min_length=2)
    last_name = serializers.CharField(max_length=255, min_length=2)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password'
                  ]

    def validate(self, attrs):
        email = attrs.get('email', '')
        if User.objects.filter(email=email).exists():
            raise serializers.ValidationError(
                {'email': ('Email is already in use')})
        return super().validate(attrs)

#     def create(self, validated_data):
#         return User.objects.create_user(**validated_data)


class LoginSerializer(serializers.ModelSerializer):
    # print("in Login Serilaizer")
    password = serializers.CharField(
        max_length=65, min_length=8, write_only=True)
    email = serializers.EmailField(max_length=255, min_length=4)

    class Meta:
        model = User
        fields = ['email', 'password']


class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, style={
                                     "input_type":   "password"})
    # password2 = serializers.CharField(
    #     style={"input_type": "password"}, write_only=True, label="Confirm password")

    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "email",
            "password",
        ]
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        first_name = validated_data["first_name"]
        email = validated_data["email"]
        password = validated_data["password"]
        last_name = validated_data["last_name"]
        print("User Model : ", User.email)
        if (email and User.objects.filter(email=email).exclude(first_name=first_name).exists()):
            raise serializers.ValidationError(
                {"email": "Email addresses must be unique."})
        # if password != password2:
        #     raise serializers.ValidationError(
        #         {"password": "The two passwords differ."})
        print("hello1")
        user = User(first_name=first_name, email=email,
                    last_name=last_name, password=password)
        # user.set_password(password)
        print("hello2")
        user.save()
        print("hello3")
        return user
