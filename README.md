# GAN on MNIST with TensorFlow

[GitHub - yihui-he/GAN-MNIST: Generative Adversarial Network for MNIST with tensorflow](https://github.com/yihui-he/GAN-MNIST)

![Untitled](https://github.com/ethanhe42/GAN-MNIST/assets/10027339/8f39f2b6-b2dd-4f0b-9fbf-f33247b0b70e)


![Untitled 1](https://github.com/ethanhe42/GAN-MNIST/assets/10027339/de4f99c4-f615-4954-9db1-e9883396dc3a)


### Tensorflow implementation

- All the codes in this project are mere replication of [Theano version](https://github.com/Newmu/dcgan_code)

### Code

- Under `face/` and `mnist/`
- model.py
- Definition of DCGAN model
- train.py
- Training the DCGAN model (and Generating samples time to time)
- util.py
- Image related utils

### Dataset

- MNIST
- http://yann.lecun.com/exdb/mnist/
- CelebA Face dataset
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Download “img_align_celeba” images
- Set “face_image_path” in train.py according to the path of downloaded dataset

### references

https://github.com/carpedm20/DCGAN-tensorflow

### Citation

If you find the code useful in your research, please consider citing:

```
@InProceedings{He_2017_ICCV,
author = {He, Yihui and Zhang, Xiangyu and Sun, Jian},
title = {Channel Pruning for Accelerating Very Deep Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
