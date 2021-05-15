#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: oberoi
file name: train_registration_dataaugmentation.py
    1. This uses E-NeuroNet model to generate the output segmentation.
    2. Uses registration based data augmentation to generate a good generalized model.
    3. A pretrained registration model is required to register the annotated dataset with 
        the unannotated dataset and generate new labels. 
    4. The following methods can be used to get a pretrained registration network:
        a) DFNet (Oberoi et. al.)
        b) VoxelMorph (Balakrishnan et. al. 2018)
usage: python train_reg_da.py --logdir=../logdir --ckdir=../models --outdir=../output --datadir=../data/image --labeldir=../data/label
--epochs=501 --batch-size=1 --cardinality=8 --reg=0.01 --learning-rate=0.0001 --classes=30 --registration-augmentation --elastic-augmentation
"""
# python imports
import os
import time
import sys
import glob
import random

# third-party imports
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import keras
import elasticdeform 

from eneuronet_model import eneuronet
from eneuronet_utils import *
sys.path.append('./registration/dfnet/src')
import data_augment


def read_flags():
    """Returns flags"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--logdir", default="../logs", help="Tensorboard log directory")
    
    parser.add_argument(
        "--ckdir", default="../models", help="Checkpoint directory")
    
    parser.add_argument(
        "--register-modelfile", default="./registration/dfnet/models_wn1_s2s", help="Pretrained registration model directory")
    
    parser.add_argument(
        "--augmentdir", default="../data/IXI-T1_FS_brainmask_crop", help="Data directory")
    
    parser.add_argument(
        "--outdir", default="../output", help="Output image directory")
    
    parser.add_argument(
        "--datadir", default="../data/image", help="Data directory")

    parser.add_argument(
        "--labeldir", default="../data/label", help="Label directory")
  
    parser.add_argument(
        "--epochs", default=501, type=int, help="Number of epochs")
    
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    
    parser.add_argument(
        "--cardinality", default=8, type=int, help="Number of branches in the building block")
    
    parser.add_argument(
        "--classes", default=30, type=int, help="Number of output segmentation classes")
    
    parser.add_argument(
        "--reg", type=float, default=0.01, help="L2 Regularizer Term")
    
    parser.add_argument(
        "--learning-rate", default=0.0001, type=float, help="Learning rate")

    parser.add_argument("--elastic-augmentation", action='store_true', default=False, help="If apply elastic deformation data augmentation")

    parser.add_argument("--registration-augmentation", action='store_true', default=False, help="If apply regstration deformation data augmentation")

    parser.add_argument(
        "--control-points", default=3, type=int, help="Number of control points for b-spline elastic deformation")
    
    parser.add_argument(
        "--std-dev", default=10, type=int, help="Standard deviation(in number of pixels) of Gaussian deformation")
    
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


def approx_dice(pred_mask, label, classes):
    """Returns a (approx) IOU score
    D = 2 * intersection / (y_pred.sum() + y_true.sum()) 
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
    """Create the network, run inference on the input batch and compute loss.
    Args:
        pred_mask (5-D array): (N, D, H, W, classes)
        label (5-D array): (N, D, H, W, 1).
    Returns:
        Pixel-wise softmax loss.
    """
    
    label_onehot = prepare_onehot_label(label, classes)
    
    ## weighted cross entropy loss
    weights = produce_weights(label_onehot, classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=pred_mask, weights=weights)
    return loss
 
    
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


def make_train_op(pred_mask, label, classes):
    """Returns a training operation
    Loss function = dice_loss(pred_mask, label)
    Args:
        pred_mask (5-D Tensor): (N, D, H, W, classes)
        label (5-D Tensor): (N, D, H, W, 1)
    Returns:
        optim: minimize operation
        loss
    """
    label= tf.stop_gradient(label,name=None)
    
    #### Different Option for losses 
    ## option1: IOU based loss
    #loss = 1-IOU_(y_pred, y_true)[0]
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits= y_pred, name= 'loss_cross_entropy'))
    ## option2: cross entropy loss
    #loss, print_op, min_freq = Loss(pred_mask, label, classes)
    ## option3: cross entropy loss + dice loss
    #loss = Loss(pred_mask, label, classes) + dice_loss(pred_mask, label, classes)
    ## option4: dice loss (as per E-NeuroNet experiments, this is sufficient)
    loss = dice_loss(pred_mask, label, classes)
    
    ## decayed learning rate
    global_step = tf.train.get_or_create_global_step()
    dec_learning_rate = tf.compat.v1.train.exponential_decay(flags.learning_rate, global_step,50000, 0.5, staircase=True)
    optim = tf.train.AdamOptimizer(learning_rate=dec_learning_rate)
    #optim = tf.train.AdamOptimizer(learning_rate=flags.learning_rate).minimize(loss, global_step=global_step)
    #optim = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss,var_list=[var for var in tf.trainable_variables()[-18:]])
    #optim =  optim.minimize(loss, global_step=global_step)
   
    ## gradient clipping for stable convergence
    gvs = optim.compute_gradients(loss)
    gvs = [(tf.clip_by_norm(grad,1), val) for grad,val in gvs]
    optim = optim.apply_gradients(gvs)
    
    return optim, loss


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


def show_params():
     """Returns the total number of trainable parameters in the model.
    """   
    total_parameters = 0
    for v in tf.trainable_variables():
        dims = v.get_shape().as_list()
        num  = int(np.prod(dims))
        total_parameters += num
        print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
    print('\n Total number of params: %d' % total_parameters)
    total_parameters_tf = tf.convert_to_tensor(total_parameters)
    return total_parameters_tf


def main(flags):
    """Main function to train the model
    Args:
        flags
    Returns:
        Trains the model
        Saves the model.ckpt in flags.ckdir directory.
        Prints the Dice Overlap and time for training. 
    """   

    ## reset the tensorflow graph    
    tf.reset_default_graph()

    ## set variables using flags
    train_img_dir = flags.datadir + '/train'
    test_img_dir =  flags.datadir + '/test'
    val_img_dir = flags.datadir + '/val'
    train_lbl_dir = flags.labeldir + '/train'
    test_lbl_dir =  flags.labeldir + '/test'
    val_lbl_dir = flags.labeldir + '/val'
    cardinality = flags.cardinality
    classes = flags.classes + 1 ## Adding 1 for background
    batch_size=flags.batch_size
    lambda_ = flags.reg
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "train", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)
    img_dir = flags.outdir    
    if os.path.exists(img_dir) == False:
        os.mkdir(img_dir)
        
        
    ## specify input shape of the image, change for a different size image
    input_shape= (160, 192, 224, 1)
    
    ## initializing placeholders for the input and output of the network
    X = tf.placeholder(tf.float32, shape=[None, 160, 192, 224, 1], name="X") ## input img may be a float
    y = tf.placeholder(tf.int32, shape=[None, 160, 192, 224, 1], name="y") ## output seg are integers
    mode = tf.placeholder(tf.bool, name="mode") 
    
    ## output segmentation masks from the eneuronet
    pred = resnext80_3d(input_shape, X, cardinality, classes)

    ## saving the variables as input and output inside tf graph               
    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    ## update the network parameters while finding train_optim and loss entropy
    with tf.control_dependencies(update_ops):
        #train_optim, loss_entropy, print_op, min_freq = make_train_op(pred, y, classes)
        train_optim, loss_entropy = make_train_op(pred, y, classes)
    with tf.control_dependencies(update_ops):
        IOU_op,update_o= IOU_(pred, y, classes)
    dice_op= approx_dice(pred, y, classes)
    
    ## saving loss, iou and dice inside the tensorflow graph
    ## these can be visualized using tensorboard
    tf.summary.scalar("LOSS", loss_entropy)
    tf.summary.scalar("IOU", IOU_op)
    tf.summary.scalar("DICE", dice_op)

    ## loading input volume names
    train_img_names, train_lbl_names  = seg_vol_list(train_img_dir, train_lbl_dir)
    n_train =  len(train_img_names)
    val_img_names, val_lbl_names  =  seg_vol_list(val_img_dir, val_lbl_dir)
    n_val =  len(val_img_names)

    ## randomly find registration image
    augment_names = glob.glob(os.path.join(flags.augmentdir, '*.nii.gz'))
    random.shuffle(augment_names)  # shuffle volume list 
    
    ## training and validation data generator
    def generator_tr():
        for i in range(len(train_img_names)):
            X_data = nib.load(train_img_names[i]).get_fdata()
            ## in case of high variation in the dataset, the data can be normalized
            #X_data_norm = ( X_data - np.min(X_data) ) / ( np.max(X_data) - np.min(X_data) ) 
            
            Y_data = nib.load(train_lbl_names[i]).get_fdata()

            ## apply registration based data augmentation  
            if(flags.registration_augmentation and (np.random.rand(1)[0]>0.667) ):
                j = np.random.randint(0, high=len(augment_names)) 
                fixed = nib.load(augment_names[j]).get_fdata()
                ## Need to change depending upon the input is normalized or not
                ## Here eneuronet does not take normalized input 
                ## but registration model used for data aug takes normalized input
                X_data, fixed = X_data/255, fixed/255 ## should be removed if using X_data_norm
                X_data, Y_data = data_augment.register(X_data, Y_data, fixed, flags.register_modelfile)
                X_data, fixed = X_data*255, fixed*255  ## should be removed if using X_data_norm
                
            #apply elastic deformation
            elif(flags.elastic_augmentation and (np.random.rand(1)[0]>0.5) ):
                [X_data, Y_data] = elasticdeform.deform_random_grid([X_data, Y_data], sigma=flags.std_dev, order=[3, 0], points=flags.control_points)

            #concat_data = np.stack((X_data_norm, Y_data), axis = -1)
            concat_data = np.stack((X_data, Y_data), axis = -1)
            yield concat_data
            
    def generator_val():
        for i in range(len(val_img_names)):
            X_data = nib.load(val_img_names[i]).get_fdata()
            #X_data_norm = ( X_data - np.min(X_data) ) / ( np.max(X_data) - np.min(X_data) ) 
            Y_data = nib.load(val_lbl_names[i]).get_fdata()
            #concat_data = np.stack((X_data_norm, Y_data), axis = -1)
            concat_data = np.stack((X_data, Y_data), axis = -1)
            yield concat_data
            
            
    train_dataset  = tf.data.Dataset.from_generator(generator_tr, output_types= tf.float32).batch(batch_size)
    val_dataset  = tf.data.Dataset.from_generator(generator_val, output_types= tf.float32).batch(batch_size)
    train_iterator  = train_dataset.make_initializable_iterator()
    val_iterator  = val_dataset.make_initializable_iterator()
    train_op = train_iterator.get_next()
    val_op  = val_iterator.get_next()
    
    ## loss of the graph
    ## different losses which can be used. Change in make_train_op also, if using different loss
    #loss_graph, print_op_graph, min_freq_graph = Loss(pred,y, classes)
    #loss_graph = Loss(pred, y, classes) + dice_loss(pred, y, classes)
    loss_graph = dice_loss(pred, y, classes)

    ## trainable parameters 
    trainable_para = show_params()
    
    summary_op = tf.summary.merge_all()

    ## allow growth of memory so that complete GPU memory is not occupied by TF
    config=tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    ## tensorflow session for training the model
    with tf.Session(config=config) as sess:
        ## update tensorboard variables
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)

        ## initialize local and global variables
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        
        ## load the tensorflow graph and weights
        print(tf.train.latest_checkpoint(flags.ckdir))
        saver = tf.train.Saver()
        if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(
                flags.ckdir):
            latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            saver.restore(sess, latest_check_point)
        else:
            try:
                os.rmdir(flags.ckdir)
            except FileNotFoundError:
                pass
            os.mkdir(flags.ckdir)
        try:
            global_step = tf.train.get_global_step(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            
            best_dice = 0
            for epoch in range(flags.epochs):   
                ## initialize train and val iterator  
                sess.run(train_iterator.initializer)
                sess.run(val_iterator.initializer)
                
                print("epoch number", epoch)
                print(sess.run(trainable_para))

                total_loss=0
                total_iou = 0
                total_dice = 0
                
                for step in range(0, n_train , flags.batch_size):
                    ## run tensorflow session for each batch to minimize the loss and update model parameters                    
                    train_np= sess.run(train_op)
                    iou_conf_mat = sess.run([update_o],feed_dict={X: train_np[:,:,:,:,0:1],
                                   y: train_np[:,:,:,:,1:2]})
                    _, loss, step_summary, global_step_value, conv_output_train, iou, dice = sess.run(
                        [train_optim, loss_entropy, summary_op, global_step, pred, IOU_op,dice_op],
                        feed_dict={X: train_np[:,:,:,:,0:1],
                                   y: train_np[:,:,:,:,1:2],
                                   mode: True})

                    ## update the summary for the global step, i.e., update loss and dice summary in graph
                    ## this summary can be viewed via tensorboard and loss graph can be plotted    
                    train_summary_writer.add_summary(step_summary,
                                                     global_step_value)
                    
                    total_loss+=loss
                    total_iou+=iou
                    total_dice+=dice
                    
                    ## save training prediction and gt after every 50 epochs
                    if (epoch % 50 == 0) and (step == 0):
                        img_header = nib.load(train_img_names[step]).header
                        out_seg = np.argmax(conv_output_train[0], axis = -1).astype('uint8')
                        lbl = nib.load(train_lbl_names[step])
                        out_seg= nib.nifti1.Nifti1Image(out_seg, None, img_header)                     
                        nib.save(out_seg, "./" + img_dir + "/" +"{}".format(epoch)+"_"+"{}".
                                    format(step)+ "_train_pred.nii.gz")      
                        nib.save(lbl, "./"+ img_dir + "/" +"{}".format(epoch)+"_"+"{}".
                                    format(step)+ "_train_label.nii.gz")
               
                avg_iou = (total_iou / n_train) *  batch_size
                avg_dice = (total_dice / n_train) *  batch_size
                avg_loss = (total_loss / n_train) * batch_size
                print('Avg Train Loss = ',avg_loss, "Avg IOU(inc. bg) = ",avg_iou, "Avg Dice(inc. bg) = ", avg_dice)
                duration = time.time() - start_time
                
                
                total_iou=0
                total_dice=0
                total_loss=0
                ## validate the model by evaluating on validation set
                for step in range(0, n_val , flags.batch_size):                    
                    val_np= sess.run(val_op)                    
                    iou_conf_mat_val = sess.run([update_o],feed_dict={X: val_np[:,:,:,:,0:1],
                                   y: val_np[:,:,:,:,1:2]})                    
                    loss_val, iou_val, dice_val, step_summary_val ,conv_output_val = sess.run(
                        [loss_graph, IOU_op, dice_op, summary_op, pred],
                        feed_dict={X: val_np[:,:,:,:,0:1],
                                   y: val_np[:,:,:,:,1:2],
                                   mode: False})

                    total_loss+=loss_val
                    total_iou+=iou_val
                    total_dice+=dice_val
                    if (epoch % 50 == 0) and (step == 0):
                        img_header = nib.load(val_img_names[step]).header
                        out_seg = np.argmax(conv_output_val[0], axis = -1).astype('uint8')
                        lbl = nib.load(val_lbl_names[step])
                        out_seg= nib.nifti1.Nifti1Image(out_seg, None, img_header)
                        nib.save(out_seg, "./" + img_dir + "/" +"{}".format(epoch)+"_"+"{}".
                                    format(step)+ "_val_pred.nii.gz")      
                        nib.save(lbl, "./" + img_dir + "/" +"{}".format(epoch)+"_"+"{}".
                                    format(step)+ "_val_label.nii.gz")    
                    
                    ## update the summary for the global step, i.e., update loss and dice summary in graph
                    ## this summary can be viewed via tensorboard and loss graph can be plotted
                    test_summary_writer.add_summary(step_summary,
                                                    (epoch + 1) * (step + 1))
                    
                
                avg_iou = (total_iou / n_val) *  batch_size
                avg_dice = (total_dice / n_val) *  batch_size
                avg_loss = (total_loss / n_val) * batch_size
                print('Avg Val Loss = ',avg_loss, 'Avg IOU(inc. bg) = ', avg_iou, "Avg Dice(inc. bg) = ", avg_dice)
                
                ## evaluate the model after every 10 epochs and save the best model
                if (epoch % 10 == 0 and avg_dice > best_dice ):
                    best_dice = avg_dice
                    #st = time.time()
                    saver.save(sess, "{}/best_model.ckpt".format(flags.ckdir))
                    #save_time = time.time() -st
                    #print("model saving time is:", save_time, "seconds. Best dice is", best_dice )
                    
               
            print('time for training  = ', duration)
            print("Best dice is :", best_dice)
            
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    flags = read_flags()
    main(flags)

