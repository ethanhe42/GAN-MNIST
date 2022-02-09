# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

![gan](https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/DCGAN.png) | ![alt tag](https://raw.githubusercontent.com/jazzsaxmafia/dcgan_tensorflow/master/mnist/vis/sample_15.jpg)
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

### references
https://github.com/carpedm20/DCGAN-tensorflow

### Citation
If you find the code useful in your research, please consider citing:

    @InProceedings{He_2017_ICCV,
    author = {He, Yihui and Zhang, Xiangyu and Sun, Jian},
    title = {Channel Pruning for Accelerating Very Deep Neural Networks},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }
