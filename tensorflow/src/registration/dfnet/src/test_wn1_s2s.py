#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
usage: python train_wn1.py --logdir=../logdir_wn1 --ckdir=../models_wn1 --imgdir=../imgdir_wn1 --datadir=../data --loss=mse
"""


import time
import os
import tensorflow as tf
import nibabel as nib
import numpy as np
import keras

from ncc3d import normalized_cross_correlation
import fileread


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--logdir", default="../logdir_wn1", help="Tensorboard log directory")

    parser.add_argument(
        "--ckdir", default="../models_wn1", help="Checkpoint directory")

    parser.add_argument(
        "--imgdir", default="../imgdir_wn1", help="Output image directory")

    parser.add_argument(
        "--datadir", default="../data", help="Data directory")

    parser.add_argument(
            "--loss", default="mse", help="mse or ncc loss")

    flags = parser.parse_args()
    return flags


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


def Loss(raw_output, label_batch):
    """
    Return the loss used for training
    Args:
      raw_output: output of the network
      label_batch: desired output
    Returns:
      loss: ncc or mse loss
    """
    
    if(flags.loss == "ncc"):
        loss = 1 - (CC(raw_output, label_batch)[0,0,0,0,0])**2
    else:
        loss = tf.losses.mean_squared_error(raw_output, label_batch)
    return loss


def main(flags):
    
    ## set variables using flags
    test_dir_f =  flags.datadir + '/test'
    test_dir_m =  flags.datadir + '/val'
    img_dir = flags.imgdir
    batch_size=1
    
    current_time = time.strftime("%m/%d/%H/%M/%S")
    test_logdir = os.path.join(flags.logdir, "test", current_time)
    
    if os.path.exists(img_dir) == False:
        os.mkdir(img_dir)
        
    ## placeholder initializing true output 
    y = tf.placeholder(tf.float32, shape=[None, 160, 192, 224, 1], name="y")

    ## loading tf graph and getting inout and output variables from it     
    saver = tf.train.import_meta_graph(flags.ckdir +"/model.ckpt.meta")   
    X = tf.get_collection("inputs")[0]
    pred = tf.get_collection("outputs")[0]

    CC_op =  CC(pred, y)
    loss_graph = Loss(pred,y) 
    config=tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    ## loading input volume names
    test_names_m, test_names_f =  fileread.read_file_sub2sub(test_dir_m, test_dir_f, batch_size)
    n_m =  len(test_names_m)
    n_f =  len(test_names_f)
    totalloss=0
    total_cc = 0
    
    for el_f in test_names_f:
        fixed = nib.load(el_f).get_fdata()
        
        ## test data generator
        def generator_test():
            for el_m in test_names_m:
                moving = nib.load(el_m).get_fdata()
                concat_data = np.stack((moving, fixed), axis = -1)
                yield concat_data    
        test_dataset  = tf.data.Dataset.from_generator(generator_test,
                                               output_types= tf.float32).batch(batch_size)
        test_iterator  = test_dataset.make_initializable_iterator()
        test_op = test_iterator.get_next()
  
        ## tensorflow session for evaluating the model    
        with tf.Session(config=config) as sess:
            ## initialize local and global variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ## load the weights from graph            
            saver.restore(sess, flags.ckdir +"/model.ckpt")
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(test_iterator.initializer)

            ## evaluate the model on test set
            for step in range(0, n_m):
                test_np= sess.run(test_op)  
                ## find loss and NCC for test set without updating model parameters             
                start_time = time.time()                  
                loss_test, cross_corr, output_test = sess.run(
                    [loss_graph, CC_op, pred],
                    feed_dict={X: test_np,
                               y: test_np[:,:,:,:,1:2] })
                duration = time.time() - start_time    
                print('Evaluation time for the image is ', duration, ' seconds.')
                totalloss+= loss_test
                total_cc+= cross_corr
    
                f = nib.load(test_names_f[step]).get_fdata()
                f_header = nib.load(test_names_f[step]).header
                f_name = test_names_f[step]
                f_name = (f_name.rsplit('/', 3))
                f_name =  (f_name[-1])
                f_name = (f_name.rsplit('.', 1))
                f_name = f_name[0]
    
                m = nib.load(test_names_m[step]).get_fdata()
                m_header = nib.load(test_names_m[step]).header
                m_name = test_names_m[step]
                m_name = (m_name.rsplit('/', 3))
                m_name =  (m_name[-1])
                m_name = (m_name.rsplit('.', 1))
                m_name = m_name[0]
    
                m_registered = output_test[0,:,:,:,0].astype('float32')
                #m_registered = (m_registered * 255).astype('int32')
                m_registered= nib.nifti1.Nifti1Image(m_registered, None, m_header)
                #f = (f * 255).astype('int32')
                f = nib.nifti1.Nifti1Image(f, None, f_header)
                #m = (m * 255).astype('int32')
                m = nib.nifti1.Nifti1Image(m, None, m_header)
    
                nib.save(m_registered, img_dir + "/" + "f_" + f_name + "m_" + m_name  + "_registered.nii") 
                nib.save(f, img_dir + "/" + f_name + "_original.nii")         
                nib.save(m, img_dir + "/" + m_name + "_original.nii")         
    
            coord.request_stop()
            coord.join(threads)
    
    avg_cc = total_cc / (n_f * n_m) 
    avg_test_loss = totalloss / (n_f * n_m)
    print('Avg Test Loss = ',avg_test_loss, 'Avg CC =', avg_cc[0,0,0,0,0])
       

if __name__ == '__main__':
    flags = read_flags()
    main(flags)
