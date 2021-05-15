#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: test.py
usage: python test.py --ckdir=../models --outdir=../output --datadir=../data/image --labeldir=../data/label --classes=30
"""
# python imports
import time
import os
import glob
import sys
import random

# third-party imports
import tensorflow as tf
import nibabel as nib
import numpy as np
import keras



def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument(
        "--ckdir", default="../models", help="Checkpoint directory")

    parser.add_argument(
        "--outdir", default="../output", help="Output image directory")

    parser.add_argument(
        "--datadir", default="../data/image", help="Data directory")
    
    parser.add_argument(
        "--labeldir", default="../data/label", help="Label directory")

    parser.add_argument(
        "--classes", default=30, type=int, help="Number of output segmentation classes without background")
    

    flags = parser.parse_args()
    return flags



def IOU_(pred_mask, label, classes):
    """Returns a (approx) IOU score

    intesection = pred_mask.flatten() * label.flatten()
    Then, IOU = intersection / (pred_mask.sum() + label.sum() - intesection)

    Args:
        y_pred (5-D array): (N, D, H, W, 1)
        y_true (5-D array): (N, D, H, W, 1)
        classes (scalar): Number of output segmentation classes
    Returns:
        float: IOU score, Confusion matrix
    """
    pred_seg = tf.argmax(pred_mask, axis = -1) #int32
    pred_flat = tf.reshape(pred_seg, [-1, ])
    label_flat = tf.reshape(label, [-1, ])
    #weights = tf.cast(tf.less_equal(gt, 4), tf.int32)
    #print("Shape of ground truth and model prediction", tf.keras.backend.int_shape(predict), tf.keras.backend.int_shape(gt) )
    iou,conf_mat = tf.metrics.mean_iou(label_flat, pred_flat, num_classes=classes)
    return iou,conf_mat

def approx_dice(pred_mask, label, classes):
    """Returns a Dice score
    D = 2 * intersection / (pred_mask.sum() + label.sum())
    Args:
        pred_mask (5-D array): (N, D, H, W, classes)
        label (5-D array): (N, D, H, W, 1).
    Returns:
        float: Dice score
    """    
    pred_seg = tf.argmax(pred_mask, axis = -1) #int32
    pred_onehot = tf.one_hot(pred_seg, depth=classes)
    pred_flat = tf.reshape(pred_onehot, [-1, classes])
    
    label_onehot = prepare_onehot_label(label, classes)
    label_flat = tf.reshape(label_onehot, [-1, classes])
    
    intersection = 2 * tf.reduce_sum(pred_flat * label_flat, axis=0)   
    denominator = tf.reduce_sum(label_flat, axis=0) + tf.reduce_sum(pred_flat, axis=0)  
    return tf.reduce_mean(intersection / denominator)


def Loss(pred_mask, label, classes):
    """
    Tensorflow Cross Entropy loss with Logits:
    Warning: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. 
    Do not call this op with the output of softmax, as it will produce incorrect results.
    Args:
        pred_mask (5-D array): (N, D, H, W, classes)
        label (5-D array): (N, D, H, W, 1).
    Returns:
        Pixel-wise softmax loss.
    """
    
    pred_flat = tf.reshape(pred_mask, [-1, classes])
    
    label_onehot = prepare_onehot_label(label, classes)
    label_flat = tf.reshape(label_onehot, [-1, classes])
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_flat, labels=label_flat, name= "softmax_loss_entropy")
    mean_loss = tf.reduce_mean(loss)
    return mean_loss
    

def prepare_onehot_label(label, classes):
    """Convert segmentation labels in onehot 
    Args:
        label (5-D array): (N, D, H, W, 1)
    Returns:
        label_onehot (5-D array): (N, D, H, W, classes)
    """
    
    label = tf.squeeze(label, squeeze_dims=[-1]) ## squeeze last dimension
    #print("*********** Prepare label input batch after squeeze", input_batch)# Reducing the channel dimension.
    label_onehot = tf.one_hot(label, depth=classes)
    #print("*********** Prepare label input batch after one hot", input_batch)
    return label_onehot
    


def decode_labels(pred_mask):
    """Decode batch of segmentation masks.
    Args:
        pred_mask (5-D array): (N, D, H, W, classes)
    
    Returns:
        pred_seg (5-D array): (N, D, H, W, 1)
    """
    pred_seg = tf.arg_max(pred_mask, -1)
    return pred_seg


def seg_vol_list(img_dir, lbl_dir):
    """Outputs the list of image name and corresponding label name
    Args:
        image directory, label directory
    Returns:
        List of image name, label name
    """    
    #volume list
    lbl_names = []
    img_names = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    random.shuffle(img_names)
    for  el in img_names:
        img_name = (el.rsplit('/', 1))[-1]
        lbl_names.append(lbl_dir+'/'+img_name)
  
    return [img_names, lbl_names]
   

def main(flags):
    """Main function to test the model
    Args:
        flags
    Returns:
        Saves the predicted segmentation at the desired location
        Prints the Dice Overlap of predicted seg with gt seg. 
    """
    
    ## set variables using flags
    test_img_dir =  flags.datadir + '/test'
    test_lbl_dir =  flags.labeldir + '/test'


    img_dir_gt = flags.outdir + '/gt'
    img_dir_pred = flags.outdir + '/pred'    
    if os.path.exists(flags.outdir) == False:
        os.mkdir(flags.outdir)    
    if os.path.exists(img_dir_gt) == False:
        os.mkdir(img_dir_gt)
    if os.path.exists(img_dir_pred) == False:
        os.mkdir(img_dir_pred)

    classes = flags.classes + 1 ## Add 1 for background        
    batch_size=1        
    
    
    ## placeholder initializing true output 
    y = tf.placeholder(tf.int32, shape=[None, 160, 192, 224, 1], name="y") ## change image size according to the image
    
    ## loading tf graph and getting inout and output variables from it     
    saver = tf.train.import_meta_graph(flags.ckdir +"/best_model.ckpt.meta")   
    X = tf.get_collection("inputs")[0]
    pred = tf.get_collection("outputs")[0]

    IOU_op,update_o= IOU_(pred, y, classes)
    loss_graph = Loss(pred, y, classes) 
    dice_op = approx_dice(pred, y, classes)

    ## loading input volume names
    test_img_names, test_lbl_names  =  seg_vol_list(test_img_dir, test_lbl_dir)
    n_test =  len(test_img_names)

    ## test data generator function
    def generator_test():
        for i in range(len(test_img_names)):
            X_data = nib.load(test_img_names[i]).get_fdata()
            Y_data = nib.load(test_lbl_names[i]).get_fdata()
            concat_data = np.stack((X_data, Y_data), axis = -1)
            yield concat_data
    
    
    test_dataset  = tf.data.Dataset.from_generator(generator_test, output_types= tf.float32).batch(batch_size)
    test_iterator  = test_dataset.make_initializable_iterator()
    test_op = test_iterator.get_next()

    ## allow growth of memory so that complete GPU memory is not occupied by TF
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    ## tensorflow session for testing the model
    with tf.Session(config=config) as sess:
        ## initialize local and global variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ## load the weights from graph        
        latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
        saver.restore(sess, latest_check_point)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        start_time = time.time()     
        totalloss=0
        total_iou = 0
        total_dice = 0
        
        ## initialize test iterator
        sess.run(test_iterator.initializer)

        ## evaluate the model on test set
        for step in range(0, n_test):
            ## find loss and IOU for test set without updating model parameters    
            test_np= sess.run(test_op) 
            iou_conf_mat = sess.run([update_o],feed_dict={X: test_np[:,:,:,:,0:1],y: test_np[:,:,:,:,1:2]})               
            start_time = time.time()
            loss_test, iou_test, dice_test, output_test = sess.run(
                [loss_graph, IOU_op, dice_op, pred],
                feed_dict={X: test_np[:,:,:,:,0:1],
                           y: test_np[:,:,:,:,1:2] })
            duration = time.time() - start_time    
            print('Evaluation time for the image is ', duration, ' seconds.')
            
            totalloss+= loss_test
            total_iou+= iou_test
            total_dice+= dice_test
            original_label = nib.load(test_lbl_names[step]).get_fdata()
            label_header = nib.load(test_lbl_names[step]).header
            subject_name = test_lbl_names[step]
            subject_name = (subject_name.rsplit('/', 3))
            subject_name =  (subject_name[-1])
            subject_name = (subject_name.rsplit('.', 3))
            subect_name = subject_name[0]
            output_test =np.argmax(output_test, axis=-1)
            predicted_label =(output_test[0]).astype('uint8')
            ## to round off: eg 61.999999999 = 62, if we dont do this, casting is making 61.999999999 = 61 which is wrong
            original_label = np.around(original_label) 
            original_label = original_label.astype('uint8')
            predicted_label= nib.nifti1.Nifti1Image(predicted_label, None, label_header)
            original_label= nib.nifti1.Nifti1Image(original_label, None, label_header)
            nib.save(original_label, img_dir_gt + "/" + subect_name + ".nii.gz")
            nib.save(predicted_label, img_dir_pred + "/" + subect_name + ".nii.gz")
            

        avg_iou = total_iou / n_test 
        avg_dice = total_dice / n_test
        avg_test_loss = totalloss / n_test
        print('Avg Test Loss = ',avg_test_loss, 'Avg IOU(inc. bg) =', avg_iou, 'Avg Dice(inc. bg) =', avg_dice)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    flags = read_flags()
    main(flags)

