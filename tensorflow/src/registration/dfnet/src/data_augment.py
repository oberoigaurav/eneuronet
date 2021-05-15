#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: data_augment.py
    1. Used to perform registration-based data augmentation proposed in E-NeuroNet
    2. We used the DFNet pre-trained weights trained over (160, 192, 224) image.
"""
# python imports
import time
import os
import sys

# third-party imports
import tensorflow as tf
import nibabel as nib
import numpy as np
import keras

from dense_image_warp_tf3d import dense_image_warp
from ncc3d import normalized_cross_correlation



def CC(img, template):
    """
    Takes a pair of images as input
    Returns a Cross Correlation score between them
    Args:
        img: registered moving image
        template: fixed image
    Returns:
        cross_corr: CC score between 0 and 1
    """
    cross_corr = normalized_cross_correlation(1, img, template)
    return cross_corr


def register(m_img, m_label, f_img, model_dir): 
    """
    Register the moving image label (labelled dataset) over fixed image (unlabelled dataset).
    Args:
        m_img: moving image
        m_label: moving image segmentation
        f_img: fixed image
        model_dir: pre-trained registration model directory
    Returns:
        m_img_out: registered moving image
        m_lbl_out: registered moving image segmentation
    """
    
    f_img = f_img[np.newaxis, :, :, :, np.newaxis]
    m_img = m_img[np.newaxis, :, :, :, np.newaxis]
    m_label = m_label[np.newaxis, :, :, :, np.newaxis]

    ## placeholder initializing input and output (registered) segmentation
    y = tf.placeholder(tf.float32, shape=[None, 160, 192, 224, 1], name="y")
    X_lbl = tf.placeholder(tf.float32, shape=[None, 160, 192, 224, 1], name="x_label")
    
    ## loading tf graph and getting input and output variables from it     
    saver = tf.train.import_meta_graph(model_dir +"/model.ckpt.meta")   
    X = tf.get_collection("inputs")[0]
    pred = tf.get_collection("outputs")[0]
    pred_flow = tf.get_collection("flow")[0]
    
    ## generating registered image and labels
    pred_flow = tf.math.round(pred_flow)
    m_img_registered = dense_image_warp(X[:,:,:,:,0:1],pred_flow)
    m_lbl_registered = dense_image_warp(X_lbl,pred_flow)    
    
    CC_op =  CC(pred, y)
    
    ## allow growth of memory so that complete GPU memory is not occupied by TF    
    config=tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True


    ## tensorflow session for evaluating the model    
    with tf.Session(config=config) as sess:
        ## initialize local and global variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ## load the weights from graph            
        saver.restore(sess, model_dir +"/model.ckpt")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ## evaluate the model on the pair of fixed and moving image
        start_time = time.time()    
        ## X = (bs, 160, 192, 224, 2), y = (bs, 160, 192, 224, 1)              
        cross_corr, output_img, output_flow, m_img_out, m_lbl_out  = sess.run(
            [CC_op, pred, pred_flow, m_img_registered, m_lbl_registered],
            feed_dict={X: np.concatenate((m_img, f_img), axis=-1), 
                       y: f_img, 
                       X_lbl: m_label}) 

        duration = time.time() - start_time    
        #print('Evaluation time for the image is ', duration, ' seconds.')
          
        m_img_out = m_img_out[0,:,:,:,0].astype('float32')
        m_lbl_out = m_lbl_out[0,:,:,:,0].astype('int32')

        coord.request_stop()
        coord.join(threads)
        
        return m_img_out, m_lbl_out

        
    
#if __name__ == '__main__':
#    model_dir = "../models_wn1"
#    f_img_file = "/home/gauravoberoi/Data/IXI-T1_FS_brainmask_vm_crop/IXI002-Guys-0828-T1_brainmask.nii.gz"
#    m_img_file = "/home/gauravoberoi/Data/mb_original_/images_mindbogg_norm/val/HLN-12-12.nii.gz"
#    m_label_file = "/home/gauravoberoi/Data/mb_original/labels_aseg_onehot/val/HLN-12-12.nii.gz"
#    f_img = nib.load(f_img_file).get_fdata()
#    m_img = nib.load(m_img_file).get_fdata()
#    m_label = nib.load(m_label_file).get_fdata()
#    m_img_header = nib.load(m_img_file).header
#    m_lbl_header = nib.load(m_label_file).header
#    f_img, m_img= f_img/255, m_img/255
#
#    #warping = "bi"
#    warped_m_img, warped_m_label  = register(m_img, m_label, f_img, model_dir)
#    
#
#    warped_m_label = nib.nifti1.Nifti1Image(warped_m_label, None, m_lbl_header)
#    warped_m_img = nib.nifti1.Nifti1Image(warped_m_img, None, m_img_header)
#    nib.save(warped_m_img, "m_img_reg.nii.gz") 
#    nib.save(warped_m_label, "m_lbl_reg.nii.gz") 
    
