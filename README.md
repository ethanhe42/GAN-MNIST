# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

![gan](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/DCGAN.png) | ![alt tag](https://github.com/jazzsaxmafia/dcgan_tensorflow/blob/master/mnist/vis/sample_15.jpg)
---|---
architecture | results

### Tensorflow implementation
  * All the codes in this project are mere replication of [Theano version](https://github.com/Newmu/dcgan_code)

### Code
 * Under face/ and mnist/
 * model.py
  * Definition of DCGAN model
 * train.py
  * Training the DCGAN model (and Generating samples time to time)
 * util.py
  * Image related utils 
 
### Dataset
 * MNIST
  * http://yann.lecun.com/exdb/mnist/
 * CelebA Face dataset 
  * http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   * Download "img_align_celeba" images
   * Set "face_image_path" in train.py according to the path of downloaded dataset
