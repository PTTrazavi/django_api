from rest_framework import serializers
from .models import music, album, Imageupload
from django.utils.timezone import now
import uuid

class ToUpperCaseCharField(serializers.CharField):
    def to_representation(self, value):
        return value.upper()

class MusicSerializer(serializers.ModelSerializer):
    days_since_created = serializers.SerializerMethodField()
    singer = ToUpperCaseCharField()

    class Meta:
        model = music
        fields = ('id', 'song', 'singer', 'last_modified_date', 'created', 'days_since_created')

    def get_days_since_created(self, obj):
        return(now() - obj.created).days

# test album
class AlbumSerializer(serializers.ModelSerializer):
    store_path = serializers.SerializerMethodField()
    result_path = serializers.SerializerMethodField()

    class Meta:
        model = album
        fields = '__all__'
        #fields = ('id', 'store_path', 'created')

    def get_store_path(self, obj):
        return(uuid.uuid1())

    def get_result_path(self, obj):
        return(uuid.uuid4())

#images
class ImageuploadSerializer(serializers.ModelSerializer):
    #result_file = serializers.SerializerMethodField()
    class Meta:
        model = Imageupload
        fields = '__all__'

    #def get_result_file(self, obj):
    #    return(obj.image_file.name.split('.')[-2] + '_matting.' + obj.image_file.name.split('.')[-1])
