#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

import ipdb

def batchnormalize(X, eps=0e-8):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / std

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / std

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[64,64,3],
            dim_z=100,
            dim_W1=1024,
            dim_W2=512,
            dim_W3=256,
            dim_W4=128,
            dim_W5=3,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_W4 = dim_W4
        self.dim_W5 = dim_W5

        self.gen_W1 = tf.Variable(tf.random_normal([dim_z, dim_W1*4*4], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([5,5,dim_W2, dim_W1], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3, dim_W2], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_W4, dim_W3], stddev=0.02), name='gen_W4')
        self.gen_W5 = tf.Variable(tf.random_normal([5,5,dim_W5, dim_W4], stddev=0.02), name='gen_W5')

        self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_W5,dim_W4], stddev=0.02), name='dim_discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W4,dim_W3], stddev=0.02), name='dim_discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2], stddev=0.02), name='dim_discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([5,5,dim_W2,dim_W1], stddev=0.02), name='dim_discrim_W4')
        self.discrim_W5 = tf.Variable(tf.random_normal([4*4*dim_W1,1], stddev=0.02), name='dim_discrim_W5')

        self.gen_params = [
                self.gen_W1,
                self.gen_W2,
                self.gen_W3,
                self.gen_W4,
                self.gen_W5
                ]
        self.discrim_params = [
                self.discrim_W1,
                self.discrim_W2,
                self.discrim_W3,
                self.discrim_W4,
                self.discrim_W5
                ]

    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        image_gen = self.generate(Z)

        p_real = self.discriminate(image_real)
        p_gen = self.discriminate(image_gen)

        discrim_cost_real = tf.nn.sigmoid_cross_entropy_with_logits( p_real, tf.ones_like(p_real) )
        discrim_cost_gen = tf.nn.sigmoid_cross_entropy_with_logits( p_gen, tf.zeros_like(p_gen) )

        discrim_cost = tf.reduce_mean(discrim_cost_real) + tf.reduce_mean(discrim_cost_gen)
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( p_gen, tf.ones_like(p_gen) ))

        return Z, image_real, discrim_cost, gen_cost

    def discriminate(self, image):
        h1 = lrelu( tf.nn.conv2d( image, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.discrim_W3, strides=[1,2,2,1], padding='SAME')) )
        h4 = lrelu( batchnormalize( tf.nn.conv2d( h3, self.discrim_W4, strides=[1,2,2,1], padding='SAME')) )
        h4 = tf.reshape(h4, [self.batch_size, -1])
        h5 = tf.matmul( h4, self.discrim_W5 )
        y = tf.nn.sigmoid(h5)
        return y

    def generate(self, Z):
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.reshape(h1, [self.batch_size,4,4,self.dim_W1])

        output_shape_l2 = [self.batch_size,8,8,self.dim_W2]
        h2 = tf.nn.deconv2d(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2) )

        output_shape_l3 = [self.batch_size,16,16,self.dim_W3]
        h3 = tf.nn.deconv2d(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )

        output_shape_l4 = [self.batch_size,32,32,self.dim_W4]
        h4 = tf.nn.deconv2d(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        h4 = tf.nn.relu( batchnormalize(h4) )

        output_shape_l5 = [self.batch_size,64,64,self.dim_W5]
        h5 = tf.nn.deconv2d(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])

        x = tf.nn.tanh(h5)
        return x

    def samples_generator(self, batch_size):

        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.reshape(h1, [batch_size,4,4,self.dim_W1])

        output_shape_l2 = [batch_size,8,8,self.dim_W2]
        h2 = tf.nn.deconv2d(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2) )

        output_shape_l3 = [batch_size,16,16,self.dim_W3]
        h3 = tf.nn.deconv2d(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )

        output_shape_l4 = [batch_size,32,32,self.dim_W4]
        h4 = tf.nn.deconv2d(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        h4 = tf.nn.relu( batchnormalize(h4) )

        output_shape_l5 = [batch_size,64,64,self.dim_W5]
        h5 = tf.nn.deconv2d(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])

        x = tf.nn.tanh(h5)
        return Z, x

