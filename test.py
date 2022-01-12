# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:15:07 2018

@author: 孙斌,熊景琦
"""
import os
import numpy as np
from PIL import Image
import click
import math
import matplotlib.pyplot as plt

from model import generator_model
from utils import load_image, deprocess_image, preprocess_image
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
def process_image(cv_img):
    #cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    #print(img.shape)
    img = (img - 127.5) / 127.5
    return img
def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False
def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]
data_path='./test_data/haze'
#all_test_path=list_image_files(image_path)

g = generator_model()
g.load_weights('generator.h5')

all_test_path=os.listdir(data_path)
for ii in all_test_path:
    img_path=data_path+"/"+ii
    test_image = np.array([process_image(load_image(img_path))])
    generated_T, generated_images = g.predict(x=test_image)
    generated = deprocess_image(generated_images)
    T = deprocess_image(generated_T)
    img = generated[0, :, :, :]
    img_T = T[0, :, :, 0]
    im = Image.fromarray(img.astype(np.uint8))
    im_T = Image.fromarray(img_T.astype(np.uint8))
    im.save('test_new/'+"dehaze_"+ii)
    im_T.save('test_T_new/'+"t"+ii)

print("done")
'''
print(all_test_path)
path=image_path+'/21.png'

test_image=np.array([process_image(load_image(path))])
generated_T, generated_images = g.predict(x=test_image)
generated = deprocess_image(generated_images)
T = deprocess_image(generated_T)
img = generated[0, :, :, :]
img_T=T[0, :, :, 0]
im = Image.fromarray(img.astype(np.uint8))
im_T = Image.fromarray(img_T.astype(np.uint8))
print(path)
print('test/'+path[22:])

im.save('test/21.png')
im_T.save('test_T/21.png')
'''
# im.save('test/'+path[22:])
# im_T.save('test_T/'+path[22:])
'''
i=1
for path in all_test_path:
    #print(path[22:])
    test_image=np.array([process_image(load_image(path))])
    #print(test_image.shape)
    generated_T, generated_images = g.predict(x=test_image)
    #print(generated_images.shape)
    #generated = np.array([deprocess_image(img) for img in generated_images])
    generated = deprocess_image(generated_images)
    T = deprocess_image(generated_T)
    #print(T.shape)
    #print(generated.shape)
    #x_test = deprocess_image(test_image)
    img = generated[0, :, :, :]
    img_T=T[0, :, :, 0]
    #output = np.concatenate((x, img), axis=1)
    im = Image.fromarray(img.astype(np.uint8))
    im_T = Image.fromarray(img_T.astype(np.uint8))
    im.save('test/'+path[22:])
    im_T.save('test_T/'+path[22:])
    i=i+1
'''    