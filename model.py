# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:38:44 2018

@author: 孙斌
"""
from keras.layers import Input, Activation, Add, UpSampling2D ,MaxPooling2D ,concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from math import ceil
from layer_utils import ReflectionPadding2D, res_block, dense_block, transition_up_block, transition_block
from layer_utils import interp_block
# the paper defined hyper-parameter:chr
channel_rate = 64
# Note the image_shape must be multiple of patch_shape
image_shape = (480, 640, 3)
input_shape = (256, 256)
patch_shape = (channel_rate, channel_rate, 3)
concat_axis=3
ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
output_nt = 1
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
input_shape_discriminator_T = (256, 256, output_nt)
n_blocks_gen = 5

def generator_model():
    """Build generator architecture."""
    # Current version : ResNet block
    '''
    inputs = Input(shape=image_shape)
    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3, 3))(x)
    
    x = Conv2D(filters=output_nt, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)

    #outputs = Add()([x, inputs])
    outputs = x
    #outputs = Lambda(lambda z: z/2)(outputs)
    '''
    #Current version : Dense block
    inputs = Input(shape=image_shape)
    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    #multi_scale:inception_v1
    #1x1
    pathway1=Conv2D(filters=16,kernel_size=(1,1),strides=1,padding='same')(x)
    #1x1->3x3
    pathway2=Conv2D(filters=32,kernel_size=(1,1),strides=1,padding='same')(x)
    pathway2=Conv2D(filters=16,kernel_size=(3,3),strides=1,padding='same')(pathway2)
    #1x1->5x5
    pathway3=Conv2D(filters=32,kernel_size=(1,1),strides=1,padding='same')(x)
    pathway3=Conv2D(filters=16,kernel_size=(5,5),strides=1,padding='same')(pathway3)
    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(x)
    pathway4=Conv2D(filters=16,kernel_size=(1,1),strides=1,padding='same')(pathway4)
    x=concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)
    #x = BatchNormalization()(x)
    #x = LeakyReLU(0.2)(x)
    #dense_block
    nb_dense_block=4
    nb_filter=64
    nb_layers_per_block=4
    nb_layers=[nb_layers_per_block]*(nb_dense_block*2+1)
    #growth_rate=16
    #compression=64
    upsampling_type='upsampling'
    dropout_rate=None
    skip_list=[x]
    for block_idx in range(nb_dense_block):
        growth_rate=16*(2**(block_idx))
        compression=64*(2**(block_idx))
        x, nb_filter = dense_block(x, nb_layers[block_idx], nb_filter, growth_rate,
                                     dropout_rate=dropout_rate)
        skip_list.append(x)
        x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter=compression
   
    _, nb_filter, concat_list =dense_block(x, nb_layers_per_block, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, return_concat_list=True)
    skip_list = skip_list[::-1] #reverse the skip list
        
    #updense_block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]
        l= concatenate(concat_list[1:],axis=concat_axis)
        t = transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type)
        x=concatenate([t,skip_list[block_idx]],axis=concat_axis)
        growth_rate=int(128*(2**(-block_idx)))
        _, nb_filter, concat_list=dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter,
                                                growth_rate, dropout_rate=dropout_rate,return_concat_list=True)
    temp=x
    #psp->conv->Tmap
    
    feature_map_size=tuple(int(ceil(image_shape[i] / 1.0)) for i in range(0,2))
    interp_block1 = interp_block(x, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(x, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(x, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(x, 6, feature_map_size, input_shape)
    

    x = concatenate([x, interp_block6, interp_block3, interp_block2, interp_block1],axis=concat_axis)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nt, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)
    
    
    #defoggy
    
    fog = concatenate([inputs, temp],axis=concat_axis)
    fog = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(fog)
    fog = BatchNormalization()(fog)
    fog = LeakyReLU(0.2)(fog)
    '''
    f_list=[fog]
    for n in range(3):
        fog = Conv2D(filters=64*(2**n), kernel_size=(3, 3), strides=1, padding='same')(fog)
        fog = BatchNormalization()(fog)
        fog = LeakyReLU(0.2)(fog)
        f_list.append(fog)
        fog=concatenate([f_list[-1],f_list[-2]])
    '''
    for i in range(n_blocks_gen):
        fog = res_block(fog, 64, use_dropout=True)
    fog = concatenate([inputs, fog],axis=concat_axis)             
    fog = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(fog)
    fog = BatchNormalization()(fog)
    fog = LeakyReLU(0.2)(fog)
    fog = ReflectionPadding2D((3, 3))(fog)
    
    fog = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(fog)
    fog = Activation('tanh')(fog)
    
    model = Model(inputs=inputs, outputs=[x,fog], name='g_dehazing_map')
    
    #model = Model(inputs=inputs, outputs=x, name='Generator')
    
    return model



def discriminator_image_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
  
    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator_T')
    return model

def discriminator_T_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=input_shape_discriminator_T)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
  
    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator_image')
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def generator_containing_discriminator_multiple_outputs(generator, discriminator, discriminator1):
    inputs = Input(shape=image_shape)
    generated_T, generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    outputs1 = discriminator1(generated_T)
    model = Model(inputs=inputs, outputs=[generated_image,outputs, generated_T, outputs1])
    #model = Model(inputs=inputs, outputs=[outputs])
    return model


if __name__ == '__main__':
   
   g = generator_model()
   g.summary()
   '''
   
   m1 = generator_containing_discriminator_multiple_outputs_T(generator_model(), discriminator_T_model())
   m1.summary()
   '''
   
   m = generator_containing_discriminator_multiple_outputs(generator_model(), discriminator_image_model(), discriminator_T_model())
   m.summary()
   
   
    
    