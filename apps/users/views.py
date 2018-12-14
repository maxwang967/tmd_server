from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework_jwt.authentication import JSONWebTokenAuthentication

from users.serializers import UserSerializer
import json


# register user for the app
class RegisterView(APIView):
    def post(self, request):
        print(request.data)
        entity = UserSerializer(data=request.data)
        if not entity.is_valid(raise_exception=False):
            return Response({"username": "nan", "password": "nan"})
        username = entity.data["username"]
        password = entity.data["password"]
        user = User.objects.create_user(username=username, password=password, email="%s@morningstarwang.com" % username)
        return Response(UserSerializer(user).data)


