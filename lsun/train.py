import ipdb
import os
import pandas as pd
import numpy as np
import lmdb
from glob import glob
from model import *
from util import *

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [64,64,3]
dim_z = 100
dim_W1 = 512#1024
dim_W2 = 256#512
dim_W3 = 128#256
dim_W4 = 64#128
dim_W5 = 3

visualize_dim=196

lsun_image_path = '/media/storage3/Study/data/lsun/images/0'
lsun_images = []
for dir,_,_ in os.walk(lsun_image_path):
    lsun_images.append(glob(os.path.join(dir, '*.jpg')))

lsun_images = lsun_images[0]
dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_W4=dim_W4,
        dim_W5=dim_W5
        )

Z_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, h_real, h_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.initialize_all_variables().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
iterations = 0
k = 2

for epoch in range(n_epochs):

    for start, end in zip(
            range(0, len(lsun_images), batch_size),
            range(batch_size, len(lsun_images), batch_size)
            ):

        batch_image_files = lsun_images[start:end]
        batch_images = map(lambda x: crop_resize( os.path.join( lsun_image_path, x) ), batch_image_files)
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        p_real_val, p_gen_val, h_real_val, h_gen_val = sess.run([p_real, p_gen, h_real, h_gen], feed_dict={Z_tf:batch_z, image_tf:batch_images})
        if np.mod( iterations, k ) != 0:
            _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        })
            print "=========== updating G =========="
            print "iteration:", iterations
            print "gen loss:", gen_loss_val

        else:
            _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        image_tf:batch_images
                        })
            print "=========== updating D =========="
            print "iteration:", iterations
            print "discrim loss:", discrim_loss_val

        print "real h:", h_real_val.mean(),"  gen h:", h_gen_val.mean()

        if np.mod(iterations, 100) == 0:
            generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample:Z_np_sample
                        })
            generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (14,14), save_path='./vis/sample_'+str(iterations/100)+'.jpg')

        iterations += 1

