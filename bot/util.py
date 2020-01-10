from PIL import Image
import sys
import requests, cv2, torch, urllib.request
import pandas as pd
from io import BytesIO
import time, datetime
from django.core.files.base import ContentFile
from .models import Imageupload
from django.shortcuts import get_object_or_404
#add frame to the image
def imgtool(pk_o): #pk of the original image
    photo_o = get_object_or_404(Imageupload, pk=pk_o)

    #get file name and extension
    f_n = photo_o.image_file.name.split("/")[-1].split(".")[0]
    f_e = photo_o.image_file.name.split('.')[-1]

    out_f_name = f_n + "_matting." + f_e
    #Load the input image

    if "https://storage.cloud" in photo_o.image_file.url: # for GCS
        response = requests.get(photo_o.image_file.url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(photo_o.image_file)
    #image process starts here
    width = 35
    # load pixels of pictures
    px = img.load()
    for x in range(0,img.size[0]):
        for y in range(0,img.size[1]):
            # add blue frame here
            if x < width or y < width or x > img.size[0] - width or y > img.size[1] - width :
                px[x,y] = 129, 216, 208 ,255

    img_io = BytesIO()
    img.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), out_f_name)
    photo_o.result_file = img_content
    photo_o.readiness = "2"
    photo_o.save()

    return photo_o.result_file.url

"""
from .deepllabv3plus2 import *
from .trimap import *
from .deep_image_matting import *
cuda = torch.cuda.is_available()
print("cuda: " + str(cuda))
deep_image_matting_model = model_dim_fn(cuda)
print("matting model loading")
from django.shortcuts import get_object_or_404

def seg_img2(photo_input):
    pk_o, pk_m, url_o, url_m = run_deeplabv3plus2(photo_input) #get seg_out pk and mask pk
    return pk_o, pk_m, url_o

def seg_matting(pk_o, pk_m, size): #pk_o, pk_m
    photo_org = get_object_or_404(Imageuploadmask, pk=pk_o)
    photo_mask = get_object_or_404(Imageuploadmask, pk=pk_m)
    title = photo_mask.title
    #see if the file is local or on GCS
    if 'http' in photo_mask.image_file.url: #GCS
        resp = urllib.request.urlopen(photo_mask.image_file.url[:])
        mask_input = np.asarray(bytearray(resp.read()), dtype="uint8")
        mask_input = cv2.imdecode(mask_input, cv2.IMREAD_GRAYSCALE)
    else:
        mask_input = photo_mask.image_file.url[1:]
        mask_input = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
    #make trimap
    trimap_input = trimap(mask_input, title, size=size, erosion=5)
    #make matting result
    if 'http' in photo_mask.image_file.url: #GCS
        result = matting_result(photo_org.image_file.url[:], trimap_input[0], title, deep_image_matting_model, cuda)
    else:
        result = matting_result(photo_org.image_file.url[1:], trimap_input[0], title, deep_image_matting_model, cuda)
    return trimap_input[1], result #url, url
"""
