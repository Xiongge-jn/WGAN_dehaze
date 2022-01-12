# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:40:49 2018

@author: 孙斌
"""

import numpy as np
from PIL import Image
import click
import matplotlib.pyplot as plt

from model import generator_model
from utils import load_image, deprocess_image, preprocess_image


def dehazing(image_path):
    data = {
        'A_paths': [image_path],
        'A': np.array([preprocess_image(load_image(image_path))])
    }
    x_test = data['A']
    g = generator_model()
    g.load_weights('generator.h5')
    
    generated_T, generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    T = np.array([deprocess_image(img) for img in generated_T])
    x_test = deprocess_image(x_test)
    for i in range(generated_images.shape[0]):
        img = T[i, :, :, 0]
        print(img.shape,type(img))
        print(img)
        im = Image.fromarray(np.uint8(img))
        im.show()
    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('dehazing'+image_path)
    '''
    generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    #T = np.array([deprocess_image(img) for img in generated_T])
    x_test = deprocess_image(x_test)
    
    for i in range(generated_images.shape[0]):
        img = generated[i, :, :, 0]
        print(img.shape,type(img))
        print(img)
        im = Image.fromarray(np.uint8(img))
        im.save('T1'+image_path)
        plt.imshow(im)
        plt.show()
    '''
    '''
    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('dehazing'+image_path)
    '''
    
@click.command()
@click.option('--image_path',default='21.png', help='Image to deblur')
def dehazing_command(image_path):
    dehazing(image_path)
    

if __name__ == "__main__":
    x=dehazing_command()
    
