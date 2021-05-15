#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: eneuronet_utils.py
General utility functions required for training.
"""

from statistics import median
import tensorflow as tf
import numpy as np
import sys 

smooth = 1e-7

def dice_loss(pred_mask, label, classes):
    """Returns Dice loss based on Dice Overlap (V-Net: Milletari et. al. 2016)
    D = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        pred_mask (5-D array): (N, D, H, W, classes)
        label (5-D array): (N, D, H, W, 1).
    Returns:
        float: Dice loss
    """
    pred_softmax = tf.nn.softmax(pred_mask)
    ## for loss calculation, direct probabilies Faustoare feed, onehot encoding is not done
    pred_flat = tf.reshape(pred_softmax, [-1, classes])
    
    label_onehot = prepare_onehot_label(label, classes)
    label_flat = tf.reshape(label_onehot, [-1, classes])
    
    smooth = 1e-7
    intersection = 2 * tf.reduce_sum(pred_flat * label_flat, axis=0) + smooth  
    denominator = tf.reduce_sum(label_flat*label_flat, axis=0) + tf.reduce_sum(pred_flat*pred_flat, axis=0) + smooth  
    dice_loss = -1 * tf.reduce_mean(intersection / denominator)
    return dice_loss


def grad(label):
    """
    Return the squared L2-norm of gradients of the flow fields
    Args:
      flow: 3 channel flow of size same as input image
    Returns:
      mean_sq_grad: mean of square of gradient of flow along all 3 dimensions 
    """
    sz = (1,  160, 192, 224, 1)    
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
    label = tf.pad(label, paddings, "CONSTANT")

    ## gradient of flow along x, y,z direction
    sq_grad_x = tf.square(label[:,2:sz[1]+2,1:sz[2]+1,1:sz[3]+1,:] - label[:,1:sz[1]+1,1:sz[2]+1,1:sz[3]+1,:])
    sq_grad_y = tf.square(label[:,1:sz[1]+1,2:sz[2]+2,1:sz[3]+1,:] - label[:,1:sz[1]+1,1:sz[2]+1,1:sz[3]+1,:])
    sq_grad_z = tf.square(label[:,1:sz[1]+1,1:sz[2]+1,2:sz[3]+2,:] - label[:,1:sz[1]+1,1:sz[2]+1,1:sz[3]+1,:])
    sq_grad = sq_grad_x + sq_grad_y + sq_grad_z
  
    #sq_grad = tf.reshape(sq_grad, [-1,])
    #mean_sq_grad = tf.reduce_mean(sq_grad) 
    #return mean_sq_grad
    return sq_grad

def get_median(v):
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])


def produce_weights(label_onehot, classes):
    """
    Generate the weight maps as specified in QuickNAT(Roy et. al. 2018)
    Args:
        label_onehot (5_d array): (N, D, H, W, classes),
    Returns:    
    """
    ## min, median and class frequency
    D, H, W = label_onehot.get_shape().as_list()[1:-1]
    class_freq = []
    for i in range(classes):
        class_freq.append( tf.reduce_sum(label_onehot[:,:,:,:,i])/(D * H * W) )
    class_freq = tf.stack(class_freq)
    class_freq = tf.reshape(class_freq, [-1])
    median_freq = get_median(class_freq) + smooth
    min_freq = tf.reduce_min(class_freq) + smooth

    ## boundary weight
    w0 = (2*(median_freq/min_freq) )
    boundary = tf.clip_by_value( grad(label_onehot) , clip_value_min=0, clip_value_max=1)
    boundary_weight = w0 * boundary
    #boundary_weight = 0

    ## class weight
    wc = (median_freq / class_freq)
    
    ## sum boundary weight and class weight
    weights = label_onehot * wc  + boundary_weight ## (1, 160, 192, 224, classes)
    ## update from: https://github.com/kwotsin/TensorFlow-ENet/blob/master/train_enet.py
    weights = tf.reduce_sum(weights, -1) ## (1, 160, 192, 224, 1)
    return weights
  
        
        
    