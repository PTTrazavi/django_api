import os
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf
import time, datetime
import requests
from io import BytesIO
from django.core.files.base import ContentFile
from .models import Imageupload
from django.shortcuts import get_object_or_404

class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513 #圖片長寬
    FROZEN_GRAPH_NAME = 'frozen' #_inference_graph
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap
def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

MODEL_xception65_trainval = DeepLabModel("model_xception65_coco_voc_trainval.tar.gz")
#MODEL_xception65_trainval = DeepLabModel(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_xception65_coco_voc_trainval.tar.gz'))
print("deeplabv3+ model loading")

def seg_result(rows, cols, cm, img):
    for x in range(0, rows):
        for y in range(0, cols):
            if cm[x][y] == 0:
                img[x][y] = np.array([255, 255, 255], dtype='uint8')
    return Image.fromarray(img)

def mask_result(rows, cols, cm, img):
    for x in range(0, rows):
        for y in range(0, cols):
            if cm[x][y] == 0:
                img[x][y] = np.array([0, 0, 0], dtype='uint8')
            else:
                img[x][y] = np.array([255, 255, 255], dtype='uint8')
    return Image.fromarray(img)

def run_deeplabv3plus2(pk_o): #output result and mask
    photo_o = get_object_or_404(Imageupload, pk=pk_o)
    #Load the input image
    if "https://storage" in photo_o.image_file.url: # for GCS
        response = requests.get(photo_o.image_file.url)
        original_im = Image.open(BytesIO(response.content))
    else:
        original_im = Image.open(photo_o.image_file)

    MODEL = MODEL_xception65_trainval

    width, height = original_im.size
    resized_im, seg_map = MODEL.run(original_im)
    cm = seg_map
    img = np.array(resized_im)
    rows = cm.shape[0]
    cols = cm.shape[1]

    img_seq = seg_result(rows, cols, cm, img)
    img_seq = img_seq.resize((width, height),Image.ANTIALIAS)
    img_mask = mask_result(rows, cols, cm, img)
    img_mask = img_mask.resize((width, height),Image.ANTIALIAS)

    #get file name and extension
    f_n = photo_o.image_file.name.split("/")[-1].split(".")[0]
    f_e = photo_o.image_file.name.split('.')[-1]

    #remove special charactor
    tbd = ['!','@','#','$','%','^','&','*','(',')','-','+','=']
    for i in tbd:
        f_n = f_n.replace(i,'')
    #if the extension is too long make it .jpg
    if len(f_e) > 7:
        f_e = "jpg"

    out_f_name = f_n + "_deeplab." + f_e
    out_m_name = f_n + "_mask." + f_e
    #save output image
    img_io = BytesIO()
    img_seq.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), out_f_name)
    photo_o.deeplab_file = img_content

    #save mask image
    img_io = BytesIO()
    img_mask.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), out_m_name)
    photo_o.mask_file = img_content
    photo_o.save()

    return pk_o #original image pk, same as input
