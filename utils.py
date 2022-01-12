# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:41:30 2018

@author: 孙斌
"""

import os
from PIL import Image
import numpy as np
import tensorflow as tf


RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths , C_paths= os.path.join(path, 'A'), os.path.join(path, 'B'), os.path.join(path,'C')
    all_A_paths, all_B_paths, all_C_paths = list_image_files(A_paths), list_image_files(B_paths), list_image_files(C_paths) 
    images_A, images_B  = [], []
    images_C=np.empty((n_images,1,256,256),dtype="float32")
    images_A_paths, images_B_paths, images_C_paths = [], [], []
    i=0
    for path_A, path_B, path_C in zip(all_A_paths, all_B_paths, all_C_paths):
        img_A, img_B, img_C = load_image(path_A), load_image(path_B), load_image(path_C)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        image_C=preprocess_image(img_C)
        images_C[i,:,:,:]=image_C
        i=i+1
        #images_C.append(preprocess_image(img_C))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        images_C_paths.append(path_C)
    
        if len(images_A) > n_images - 1: break
    images_C = images_C.reshape(n_images,256,256,1)
    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths),
        'C': images_C,
        'C_paths': np.array(images_C_paths)
    }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
if __name__ == "__main__":
    data = load_images('dataset/train_data', 10)
    t_train, y_train, x_train = data['C'], data['B'], data['A']
    print(t_train.shape, y_train.shape, x_train.shape)
    print()