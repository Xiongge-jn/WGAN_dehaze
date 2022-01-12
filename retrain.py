# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:10:08 2019

@author: 515
"""

import os
import datetime
import click
import numpy as np
import tqdm
import sys
import keras.backend as K
from utils import load_images, write_log
from losses import wasserstein_loss, perceptual_loss, MSE
from model import generator_model, discriminator_T_model, discriminator_image_model,generator_containing_discriminator_multiple_outputs
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, RMSprop
BASE_DIR = 'weights_Gan_retrain/'

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

def save_all_weight(d,dt,g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_image_{}.h5'.format(epoch_number)), True)
    dt.save_weights(os.path.join(save_dir, 'discriminator_T_{}.h5'.format(epoch_number)), True)

def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    print("数据加载中：")
    data = load_images('dataset/train_data', n_images)
    t_train, y_train, x_train = data['C'], data['B'], data['A']
    print("数据加载完成！")
    g = generator_model()
    g.load_weights('generator.h5')
    d = discriminator_image_model()
    d.load_weights('discriminator_image.h5')
    dt = discriminator_T_model()
    dt.load_weights('discriminator_T.h5')
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d ,dt)
    #learning_rate/10, step=50
    d_opt = RMSprop(lr=0.00001)
    dt_opt = RMSprop(lr=0.00001)
    #g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = RMSprop(lr=0.00001)
    #single CNN
    '''
    g.compile(optimizer=g_opt, loss=MSE)
    g_losses = []
    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_fog_batch = x_train[batch_indexes]
            image_t_batch = t_train[batch_indexes]
            g_loss=g.train_on_batch(image_fog_batch,image_t_batch)
            g_losses.append(g_loss)
    
        print(np.mean(g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} \n'.format(epoch, np.mean(g_losses)))
            
        save_all_weight(g, epoch, int(np.mean(g_losses)))
    '''
    '''
    #generate T by using WGAN
    #compile models    
    d.trainable=True
    d.compile(loss=wasserstein_loss, optimizer=d_opt)
    #g.compile(loss=MSE,optimizer=d_on_g_opt)
    d.trainable=False
    d_on_g.compile(loss=[MSE, wasserstein_loss],loss_weights=[100,1], optimizer=d_on_g_opt)
    d.trainable=True
    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])
        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_fog_batch = x_train[batch_indexes]
            image_t_batch = t_train[batch_indexes]
            generated_images = g.predict(x=image_fog_batch, batch_size=batch_size)
            for _ in range(critic_updates):
                #clip discriminator weights
                for l in d.layers:
                    weights=l.get_weights()
                    weights=[np.clip(w,-0.01,0.01) for w in weights]
                    l.set_weights(weights)
                d_loss_real = d.train_on_batch(image_t_batch, -np.ones((batch_size, 1)))
                d_loss_fake = d.train_on_batch(generated_images, np.ones((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
            d.trainable=False
            d_on_g_loss=d_on_g.train_on_batch(image_fog_batch,[image_t_batch,-np.ones((batch_size, 1))])
            d_on_g_losses.append(d_on_g_loss)
            d.trainable=True
        print(np.mean(d_losses), np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))
    '''
    #generate T by using WGAN
    #compile models    
    d.trainable=True
    d.compile(loss=wasserstein_loss, optimizer=d_opt)
    dt.trainable=True
    dt.compile(loss=wasserstein_loss, optimizer=dt_opt)
    #g.compile(loss=MSE,optimizer=d_on_g_opt)
    d.trainable=False
    dt.trainable=False
    d_on_g.compile(loss=[perceptual_loss, wasserstein_loss, MSE, wasserstein_loss],loss_weights=[100, 1, 100, 1], optimizer=d_on_g_opt)
    d.trainable=True
    dt.trainable=True
    for epoch in tqdm.tqdm(range(epoch_num)):
        #if epoch>0:
        #   K.set_value(RMSprop.lr , 0.1*K.get_value(RMSprop.lr))
        permutated_indexes = np.random.permutation(x_train.shape[0])
        d_losses = []
        dt_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_fog_batch = x_train[batch_indexes]
            image_clean_batch = y_train[batch_indexes]
            image_t_batch = t_train[batch_indexes]
            generated_T, generated_images = g.predict(x=image_fog_batch, batch_size=batch_size)
            for _ in range(critic_updates):
                #clip discriminator weights
                for l in d.layers:
                    weights=l.get_weights()
                    weights=[np.clip(w,-0.01,0.01) for w in weights]
                    l.set_weights(weights)
                d_loss_real = d.train_on_batch(image_clean_batch, -np.ones((batch_size, 1)))
                d_loss_fake = d.train_on_batch(generated_images, np.ones((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
                for lt in dt.layers:
                    weights=lt.get_weights()
                    weights=[np.clip(w,-0.01,0.01) for w in weights]
                    lt.set_weights(weights)
                dt_loss_real = dt.train_on_batch(image_t_batch, -np.ones((batch_size, 1)))
                dt_loss_fake = dt.train_on_batch(generated_T, np.ones((batch_size, 1)))
                dt_loss = 0.5 * np.add(dt_loss_fake, dt_loss_real)
                dt_losses.append(dt_loss)
            d.trainable=False
            dt.trainable=False
            #train g
            d_on_g_loss=d_on_g.train_on_batch(image_fog_batch,[image_clean_batch, -np.ones((batch_size, 1)), image_t_batch, -np.ones((batch_size, 1))])
            d_on_g_losses.append(d_on_g_loss)
            d.trainable=True
            dt.trainable=True
        print(np.mean(d_losses), np.mean(dt_losses), np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(dt_losses), np.mean(d_on_g_losses)))
        save_all_weight(d, dt, g, epoch, int(np.mean(d_on_g_losses)))
        
@click.command()
@click.option('--n_images', default=4800, help='Number of images to load for training')
@click.option('--batch_size', default=2 , help='Size of batch')
@click.option('--log_dir', default='1og/',required=True, help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=50, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()