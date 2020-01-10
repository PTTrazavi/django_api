from django.db import models
from django.core.files import File
import os

# Create your models here.
class music(models.Model):
    song = models.CharField(max_length=64)
    singer = models.CharField(max_length=32)
    last_modified_date = models.DateTimeField(auto_now = True)
    created = models.DateTimeField(auto_now_add = True)

class album(models.Model):
    store_path = models.CharField(max_length=64)
    result_path = models.CharField(max_length=64)
    readiness = models.CharField(max_length=4)
    last_modified_date = models.DateTimeField(auto_now = True)
    created = models.DateTimeField(auto_now_add = True)


#from django.core.files.storage import default_storage #GCS
class Imageupload(models.Model):
    image_file = models.ImageField(upload_to='images/')
    #result_file = models.TextField()
    result_file = models.ImageField(upload_to='images/', blank=True)
    readiness = models.CharField(max_length=1, default='0')
    date_of_upload = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.image_file.name
