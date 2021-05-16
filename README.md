# E-NeuroNet: Efficient 3D CNN for Segmentation of Neuroimages

## Highlights
E-NeuroNet is an efficient fully three-dimensional (3D) CNN based deep learning model for fast
and accurate segmentation of MR images of the Brain. Some of the highlights of this work are :

(I). The first fully 3D convolution architecture for brain parcellation, which is able to learn
from complete 3D geometry of the brain.

(II). 3D multi-branch residual learning architecture 50× faster (taking less than a second)
on a GPU and 300× faster (taking less than a minute) on a CPU, when compared to the
baseline brain parcellation methods.

(III). Reduced memory requirement by more than 2× compared to previous fully 3D convolution methods. The memory requirement is the same as that for the 2D CNN based methods
which enables training on any standard GPU.

(IV). Additional regularization leading to superior generalization on unseen datasets of different age distribution and disease conditions, without additional fine-tuning.

(V). Novel registration-based data augmentation technique which leverages the use of unannotated dataset during training, reducing the dependence on high fidelity annotated data
and allowing the network to learn using a limited quantity of ground truth.

## Abstract

In this work, we propose a novel fully 3D CNN architecture, the E-NeuroNet, with reduced memory, computational, and annotated data requirements. This network utilizes a deep residual learning-based method, via 3D encoder-decoder extension to the well known 2D encoder architecture of ResNeXt. It also incorporates summation of coarse level skip connection features to the fine level features instead of concatenation or max-pooling. These lead to significantly fewer trainable parameters while enabling contextual information along all three dimensions of the image. We show that the network requires only 17.2GB of memory and can produce parcellation under 1 second on a standard GPU and under 1 minute on a standard CPU.

We also demonstrate that the E-NeuroNet has superior generalization and robustness to image noise, contrast, resolution, sample age group and distortion in the brain structures as compared to slice-based counterparts. We address the challenge of the requirement of high quality annotated data by deploying a data augmentation strategy based on image registration and show that it performs better than the standard elastic deformation based augmentation strategy.
![alt text](https://github.com/oberoigaurav/eneuronet/blob/master/tensorflow/diagrams/eneuronet_graphical_abstract.png?raw=true)

## E-NeuroNet Architecture

We propose E-NeuroNet, an efficient residual learning architecture with multiple parallel branches. The architecture configuration is shown below:

• 80 layer encoder-decoder architecture

• Fully convolution encoder, stride 2 for downsampling

• Fully deconvolution decoder, stride 2 for upsampling

• Cardinality of 8

• Layer Dropout of 0.8

• No batch normalization

• LeakyReLU activation with a rate of 0.3

![alt text](https://github.com/oberoigaurav/eneuronet/blob/master/tensorflow/diagrams/eneuronet_architecture.png?raw=true)

## Methodology

### Overview of Methodology

We parameterize the segmentation label probability maps using the E-NeuroNet architecture and use the random elastic deformation data augmentation technique to incorporate variance into the model. This method generates synthetic deformations in the training dataset, which are correlated to original images, and thus provides limited variance in terms of data augmentation.

![alt text](https://github.com/oberoigaurav/eneuronet/blob/master/tensorflow/diagrams/methodology_elastic_da.png?raw=true)

### Overview of Methodology in Annotation Efficiency Setting

Here, we leverage the use of unannotated dataset while training E-NeuroNet with smaller high quality labelled dataset. Unannotated dataset was utilized for the novel registration-based data augmentation, where we randomly register a labelled image to an image in an unlabelled dataset for producing uncorrelated natural deformations in the labelled dataset.

![alt text](https://github.com/oberoigaurav/eneuronet/blob/master/tensorflow/diagrams/methodology_registration_da.png?raw=true)


## DataSet Used 

### Training 
We utilized the publicly available Mindboggle-101 dataset for training and evaluation. Mindboggle is a collection of 101 T1-weighted human brain MR Images, available along with FreeSurfer FreeSurfer cortical parcellation and manually edited cortical parcellation. Mindboggle-101 dataset can be found at https://mindboggle.info/data.html.

### Data Augmentation
We leveraged the use of unlabelled IXI dataset while training to improve the generalization performance of our method. In the annotation efficiency setting, E-NeuroNet was trained only on 41 Mindboggle labelled images to give good accuracy. The dataset is available at https://brain-development.org/ixi-dataset/.

### Evaluation
We used the following datasets for evaluating the generalization performance of our method. 

#### ABIDE II
ABIDE II is available at http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html.

#### ADNI
ADNI is available at http://adni.loni.usc.edu.

#### CANDI
CANDI is available at https://www.nitrc.org/projects/candi_share/.

#### IBSR
The dataset is provided by the Center for Morphometric Analysis at Massachusetts General
Hospital and is freely available at http://www.cma.mgh.harvard.edu/ibsr/ and https://www.nitrc.org/projects/ibsr/.

#### BraTS
The dataset is freely available at https://www.med.upenn.edu/cbica/brats2020/. 

## References
E-NeuroNet, Oberoi et. al. 2021
