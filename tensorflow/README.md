# E-NeuroNet: Efficient 3D CNN for Segmentation of Neuroimages

E-NeuroNet is an efficient fully three-dimensional (3D) CNN based deep learning model for fast
and accurate segmentation of MR images of Brain. Some of the highlights of this work are :

(I). The first fully 3D convolution architecture for brain parcellation, which is able to learn
from complete 3D geometry of the brain.

(II). 3D multi-branch residual learning architecture 50× faster (taking less than a second)
on a GPU and 300× faster (taking less than a minute) on a CPU, when compared to the
baseline brain parcellation methods.

(III). Reduced memory requirement by more than 2× compared to previous fully 3D convo-
lution methods. The memory requirement is same as that for the 2D CNN based methods
which enables training on any standard GPU.

(IV). Additional regularization leading to superior generalization on unseen datasets of dif-
ferent age distribution and disease conditions, without additional fine-tuning.

(V). Novel registration-based data augmentation technique which leverages the use of unan-
notated dataset during training, reducing the dependence on high fidelity annotated data
and allowing the network to learn using limited quantity of ground truth.

![alt text](https://github.com/oberoigaurav/eneuronet/blob/master/diagrams/eneuronet_graphical_abstract.pdf?raw=true)
