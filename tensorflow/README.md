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

We trained the model on the MR scans normalized by FreeSurfer. Specifically, we used norm.mgz for training the model which can be generating using FreeSurfer's autorecon1 pipeline as shown below.


	export SUBJECTS_DIR=/home/subjectdir
	export FREESURFER_HOME=/usr/local/freesurfer
	source $FREESURFER_HOME/SetUpFreeSurfer.sh
	## autorecon1
	recon-all -s imagename -i image.mgz -autorecon1 -parallel

We then cropped the norm.mgz of size (256, 256, 256) to (160, 192, 224) which saves the GPU usage by almost 60%.

## Training 

### Elastic Deformation Based Data Augmentation
The model can be trained using train.py as shown below:

	python train.py --logdir=../logdir --ckdir=../models --outdir=../output --datadir=../data/image --labeldir=../data/label--epochs=501 --batch-size=1 --cardinality=8 --reg=0.01 --learning-rate=0.0001 --classes=30 --data-augmentation --control-points=2 --std-dev=15

#### Args
logdir: Tensorboard log directory

ckdir: Checkpoint directory

outdir: Output image directory, where prediction segmention will be saved

datadir: Data directory

labledir: Label directory

epochs: Number of epochs

batch-size: Batch size (default=1)

cardinality: Number of parallel branches in the building block of architecture

classes: Number of segmentation classes (excluding background)

reg: Constant of CNN Kernal weight regularization

learning-rate: Learning rate for the Optimization algorithm

data-augmentation: If apply elastic deformation data augmentation

control-points: Number of control points for b-spline elastic deformation

std-dev: Standard deviation(in number of pixels) of Gaussian deformation

 
### Registration Based Data Augmentation (Annotation Efficient Setting)

	python train_reg_da.py --logdir=../logdir --ckdir=../models --outdir=../output --datadir=../data/image --labeldir=../data/label --epochs=501 --batch-size=1 --cardinality=8 --reg=0.01 --learning-rate=0.0001 --classes=30 --registration-augmentation --elastic-augmentation

#### Args
registration-augmentation: If apply registration-based deformation data augmentation

elastic-augmentation: If apply elastic deformation data augmentation

For training this model, we require a pre-trained learing based deformable registration model. This will be utilized for deformable registration based data-augmentation while training the E-NeuroNet architecture. 

We provide a pre-trained DFNet model for deformable registration, which is trained over norm.mgz from Mindboggle dataset, cropped to size (160, 192, 224). If training is required over some other dataset type, the following methods may be used:

a) DFNet: http://cds.iisc.ac.in/faculty/yalavarthy/dfnet/DFNetWarpNet.pdf, https://github.com/oberoigaurav/warpnet

b) VoxelMorph: 
	Balakrishnan, Guha, et al. "VoxelMorph: a learning framework for deformable medical image registration." IEEE transactions on medical imaging 38.8 (2019): 1788-1800.
	
## Testing 

The model can be tested using test.py as shown below:
	python test.py --ckdir=../models --outdir=../output --datadir=
	../data/image --labeldir=../data/label --classes=30

## Evaluation

The performance of model can be evaluated using EvaluateSegmentation tool which is freely available at https://github.com/Visceral-Project/EvaluateSegmentation.
