#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: ncc3d.py
Normalized Cross Correlation (NCC) of 5D tensors in TensorFlow
Adapted from Weiye Li (liwe@student.ethz.ch) modification of:
    https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
"""


import numpy as np
import tensorflow as tf
import nibabel as nib


def ncc3d(img, template, template_numel, strides=[1,1,1,1,1], padding='VALID', epsilon=0.00000001):
    """
    perform 3D NCC
    Inputs: img: larger image of shape [batch=1, depth, height, width, channel=1]
            template: template of shape [depth, height, width, in_channel=1, out_channel=1]
            
    		template_numel: depth * height * width
    """

    # subtract template and image mean across [height, width] dimension
    template_zm = template - tf.reduce_mean(template, axis=[0,1,2], keep_dims=True)
    img_zm = img - tf.reduce_mean(img, axis=[1,2,3], keep_dims=True)

    # define conv function
    conv = lambda x, y: tf.nn.conv3d(x, y, padding=padding, strides=strides)
    # compute template variance
    template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1,2], keep_dims=True) + epsilon 
    # compute local image variance
    ones = tf.ones_like(template)
    localsum_sq = conv(tf.square(img), ones)
    localsum = conv(img, ones)
    img_var = localsum_sq - tf.square(localsum)/template_numel + epsilon 
    # Remove small machine precision errors after subtraction
    img_var = tf.where(img_var<0, tf.zeros_like(img_var), img_var)
    # compute 3D NCC
    denominator = tf.sqrt(template_var * img_var)
    numerator = conv(img_zm, template_zm)
    out = tf.div(numerator, denominator)
    # Remove any NaN in final output
    out = tf.where(tf.is_nan(out), tf.zeros_like(out), out)

    return out


def normalized_cross_correlation(batch_sz, img, template):
	"""
	NCC of 5D tensor per batch and channel dimension
	Inputs: img: larger image of shape [batch, depth, height, width, channel]
	        template: template of shape [batch, depth, height, width, channel]
	NOTE: img and template have the same batch and channel size
	"""

	i_shape = img.get_shape().as_list()
	t_shape = template.get_shape().as_list()
	B = i_shape[0] # number of batches
	# param. used for slicing img and template into batch * channel 3D slices
	num_slices_template = batch_sz
	num_slices_img = batch_sz
	assert(num_slices_template == num_slices_img)
	# transpose and shape
	template = tf.transpose(template, perm=[1,2,3,0,4]) # [D,H,W,B,C]
	img = tf.transpose(img, perm=[0,4,1,2,3]) # [B,C,D,H,W]
	Ht, Wt, Dt, Bt, Ct = tf.unstack(tf.shape(template))
	Bi, Ci, Hi, Wi, Di = tf.unstack(tf.shape(img))
	template = tf.reshape(template, [Ht, Wt, Dt, 1, Bt*Ct])
	img = tf.reshape(img, [Bi*Ci, Hi, Wi, Di, 1])
	# get slices per channel per batch
	template_slices = tf.split(value=template, num_or_size_splits=num_slices_template, axis=4)
	img_slices = tf.split(value=img, num_or_size_splits=num_slices_img, axis=0)
	# slice-wise NCC
	nt = int(t_shape[1]*t_shape[2]*t_shape[3])
	ncc_slices = [ncc3d(x, y, template_numel=nt) for x, y in zip(img_slices, template_slices)]
	# adjust final dimension
	ncc_out = tf.concat(values=ncc_slices, axis=4) # [1,Dout,Hout,Wout,B*C]
	ncc_out = tf.concat(tf.split(ncc_out, batch_sz, axis=4), axis=0) # [B,Dout,Hout,Wout,C]
	ncc_out = tf.expand_dims(tf.reduce_mean(ncc_out, axis=4), axis=4) # [B,Dout,Hout,Wout,Dout,1]

	return ncc_out

