#-*- coding: utf-8 -*-
import tensorflow as tf
import ipdb

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -(t * tf.log(o) + (1.- t)*tf.log(1. - o))

class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[28,28,1],
            dim_z=100,
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1+dim_y, dim_W2*7*7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2*7*7+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')

    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        image_gen = self.generate(Z,Y)

        p_real = self.discriminate(image_real, Y)
        p_gen = self.discriminate(image_gen, Y)

        discrim_cost_real = bce(p_real, tf.ones_like(p_real))
        discrim_cost_gen = bce(p_gen, tf.zeros_like(p_gen))
        discrim_cost = tf.reduce_mean(discrim_cost_real) + tf.reduce_mean(discrim_cost_gen)

        gen_cost = tf.reduce_mean(bce( p_gen, tf.ones_like(p_gen) ))

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.pack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat(3, [image, yb*tf.ones([self.batch_size, 28, 28, self.dim_y])])

        h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h1 = tf.concat(3, [h1, yb*tf.ones([self.batch_size, 14, 14, self.dim_y])])

        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat(1, [h2, Y])

        h3 = lrelu( batchnormalize( tf.matmul(h2, self.discrim_W3 ) ))
        h3 = tf.concat(1, [h3, Y])
        y = tf.nn.sigmoid(h3)
        return y

    def generate(self, Z, Y):

        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat(1, [Z,Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat(1, [h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size,7,7,self.dim_W2])
        h2 = tf.concat( 3, [h2, yb*tf.ones([self.batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [self.batch_size,14,14,self.dim_W3]
        h3 = tf.nn.deconv2d(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat( 3, [h3, yb*tf.ones([self.batch_size, 14,14,self.dim_y])] )

        output_shape_l4 = [self.batch_size,28,28,self.dim_channel]
        h4 = tf.nn.deconv2d(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return x

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat(1, [Z,Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat(1, [h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,7,7,self.dim_W2])
        h2 = tf.concat( 3, [h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [batch_size,14,14,self.dim_W3]
        h3 = tf.nn.deconv2d(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat( 3, [h3, yb*tf.ones([batch_size, 14,14,self.dim_y])] )

        output_shape_l4 = [batch_size,28,28,self.dim_channel]
        h4 = tf.nn.deconv2d(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z,Y,x


