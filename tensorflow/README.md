# E-NeuroNet: Efficient 3D CNN for Segmentation of Neuroimages

## Introduction
A Tensorflow 1.X implementation of E-NeuroNet. The model is developed using Keras and Tranined using Tensorflow. 

## Dependencies
The following dependencies/libraries were used to train and test the model.

TensorFlow==1.3.1 (TensorFlow 1.x)

Keras 2.3.1

Nibabel 3.0.0

Numpy 1.17.3

Elasticdeform 0.4.6

## Preprocessing

We trained the model on the MR scans normalized by FreeSurfer. Specifically, we used norm.mgz for training the model. 

export SUBJECTS_DIR=

export FREESURFER_HOME=/usr/local/freesurfer

source $FREESURFER_HOME/SetUpFreeSurfer.sh

recon-all -s imagename -i image.mgz -autorecon1 -parallel


## Training 

## Testing 

## Evaluation
