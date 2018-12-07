from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth.models import User
from users.serializers import UserSerializer


# register user for the app
class RegisterView(APIView):
    def post(self, request):
        entity = UserSerializer(data=request.data)
        entity.is_valid(raise_exception=True)
        username = entity.data["username"]
        password = entity.data["password"]
        user = User.objects.create_user(username=username, password=password, email="%s@morningstarwang.com" % username)
        if user:
            return Response(UserSerializer(user).data)
        else:
            return Response({})


# predict the transportation mode using PyTorch model
class PredictView(APIView):
    def post(self, request):
        request.data
