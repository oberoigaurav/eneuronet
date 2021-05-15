#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: eneuronet_model.py
    1. 80 layer  multibranch encoder-decoder residual architecture.
    2. 3D convolutions and deconvolutions for upsampling and downsampling.
    3. Summation based skip connections.
    4. Kernal weight regularization and dropout.
"""

import tensorflow as tf 
import keras
from keras import layers
from keras.layers.core import Layer
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Average, BatchNormalization, ReLU, LeakyReLU, ZeroPadding3D, Softmax, Dropout, UpSampling2D, Activation 
from keras.activations import tanh
from keras.regularizers import l2

def eneuronet(input_shape, input_tensor, cardinality, classes):
    """
    Implementation of Efficient NeuroNet Architecture of Segmentation (E-NeuroNet)
    Takes input as the 3D image.
    Returns the "classes" channel output segmentation masks size same as input image.
    Args:
        input_shape (3-D Tensor): Shape of input image: (H, W, C)
        input_tensor (3-D Tensor): Image tensor of (H, W, C)
        cardinality(scalar): Number of parallel paths (same as in ResNeXt module)
        classes(scalar): Number of segmentation classes

    Returns:
        c8: "classes" channel segmentation masks of size same as input image
    """

    weights= None
    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) ')
    if input_shape is None:
        raise ValueError('Please provide a valid input_shape')
    if input_tensor is None:
        raise ValueError('Please provide a valid input_tensor')
    else:
        x = input_tensor    
        
        
    def add_common_layers(y):
        ## batch normalization not required, as we use batch size = 1
        #y = BatchNormalization()(y)         
        y = LeakyReLU(alpha=0.3)(y)
        y = Dropout(0.8)(y)
        return y

    def encoding_grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return Conv3D(nb_channels, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, :, j * _d:j * _d + _d])(y)
            groups.append(Conv3D(_d, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)
        return y

    def decoding_grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return Conv3DTranspose(nb_channels, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, :, j * _d:j * _d + _d])(y)
            groups.append(Conv3DTranspose(_d, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)
        return y
    
    def encoding_residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1, 1), _project_shortcut=False):
        """
        residual block for encoder
        batch normalization not required, as batch size = 1
        """
        shortcut = y
        #nb_channels_in = int(nb_channels_in/4)

        # 1st layer with 1X1X1 convolution in residual block
        y = Conv3D(nb_channels_in, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        # 2nd layer with 3X3X3 convolution in residual block
        y = encoding_grouped_convolution(y, nb_channels_in , _strides=_strides)
        y = add_common_layers(y)

        # 3nd layer with 1X1X1 convolution in residual block
        y = Conv3D(nb_channels_out, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(y)
        #y = BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1, 1):
            # when the dimensions changes, projection shortcut is used to match dimensions(done by 1X1X1 convolution)
            shortcut = Conv3D(nb_channels_out, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(shortcut)
            #shortcut = BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = LeakyReLU(alpha=0.3)(y)
        return y
    
    def decoding_residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1, 1), _project_shortcut=False):
        """
        residual block for decoder
        batch normalization not required, as batch size = 1
        """
        shortcut = y
        #nb_channels_in = int(nb_channels_in/4)

        # 1st layer with 1X1X1 deconvolution in residual block
        y = Conv3DTranspose(nb_channels_in, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        # 2nd layer with 3X3X3 deconvolution in residual block
        y = decoding_grouped_convolution(y, nb_channels_out, _strides=_strides)
        y = add_common_layers(y)

        # 3nd layer with 1X1X1 deconvolution in residual block
        y = Conv3DTranspose(nb_channels_out, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(y)
        #y = BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1, 1):
            # when the dimensions changes, projection shortcut is used to match dimensions(done by 1X1X1 deconvolution)
            shortcut = Conv3DTranspose(nb_channels_out, kernel_size=(1, 1, 1), kernel_regularizer=l2(0.001), strides=_strides, padding='same')(shortcut)
            #shortcut = BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = LeakyReLU(alpha=0.3)(y)
        return y
            
    # conv1, input_img_sz=1, output_img_sz=1/2
    c1 = Conv3D(16, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=(2, 2, 2), padding='same')(x)
    c1 = Conv3D(32, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(c1)
    c1 = add_common_layers(c1)
    
    # conv2, input_img_sz=1/2, output_img_sz=1/4
    c2 = c1
    c2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(c2)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        c2 = encoding_residual_block(c2, 32, 64, _project_shortcut=project_shortcut)
        
    # conv3, input_img_sz=1/4, output_img_sz=1/8
    c3 = c2
    for i in range(4):
        # down-sampling is performed by conv3_1 and conv4_1 with a stride of 2
        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
        c3 = encoding_residual_block(c3, 64, 128, _strides=strides)

    # conv4, input_img_sz=1/8, output_img_sz=1/16
    c4 = c3
    for i in range(6):
        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
        c4 = encoding_residual_block(c4, 128, 256, _strides=strides)

    # conv 5(deconv), input_img_sz=1/16, output_img_sz=1/8
    c5 = c4
    for i in range(6):
        # up-sampling is performed by conv5_1, conv6_1 and conv7_1 with a stride of 2
        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
        c5 = decoding_residual_block(c5, 256, 128, _strides=strides)   
    
    # conv 6(deconv), input_img_sz=1/8, output_img_sz=1/4
    c6 = c5 + c3
    for i in range(4):
        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
        c6 = decoding_residual_block(c6, 128, 64, _strides=strides) 

    # conv 7(deconv), input_img_sz=1/4, output_img_sz=1/2
    c7 = c6 + c2
    for i in range(3):
        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
        c7 = decoding_residual_block(c7, 64, 32, _strides=strides)
        
    # conv8(deconv), input_img_sz=1/2, output_img_sz=1
    c8 = c7 + c1 
    c8 = Conv3DTranspose(32, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=(2, 2, 2), padding='same')(c8)
    c8 = Conv3D(classes, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), strides=(1, 1, 1), padding='same')(c8)
    c8 = add_common_layers(c8)
    ## Softmax is not required here to convert into probability, as it is applied by tf in the loss function
    c8 = Conv3D(classes, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(c8) 

    return c8


