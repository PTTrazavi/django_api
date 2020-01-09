from django.shortcuts import render, get_object_or_404
from .models import music, album, Imageupload
from .serializers import MusicSerializer, AlbumSerializer, ImageuploadSerializer
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView

# Create your views here.
class MusicViewSet(viewsets.ModelViewSet):
    queryset = music.objects.all()
    serializer_class = MusicSerializer

class AlbumViewSet(viewsets.ModelViewSet):
    queryset = album.objects.all()
    serializer_class = AlbumSerializer

class ImageuploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        file_serializer = ImageuploadSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
