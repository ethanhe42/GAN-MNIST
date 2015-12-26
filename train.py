import ipdb
import os
import pandas as pd
import numpy as np
from model import *
from util import *

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [64,64,3]
dim_z = 100
dim_W1 = 1024
dim_W2 = 512
dim_W3 = 256
dim_W4 = 128
dim_W5 = 3

visualize_dim=196

face_image_path = '/media/storage3/Study/data/celeb/img_align_celeba'
face_images = filter(lambda x: x.endswith('jpg'), os.listdir(face_image_path))

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

Z_tf, image_tf, d_cost_tf, g_cost_tf = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

adam_optimizer = tf.train.AdamOptimizer(learning_rate)

grad_var_discrim = adam_optimizer.compute_gradients(d_cost_tf, dcgan_model.discrim_params)
grad_var_gen = adam_optimizer.compute_gradients(d_cost_tf, dcgan_model.gen_params)

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

train_op_discrim = adam_optimizer.apply_gradients(grad_var_discrim)
train_op_gen = adam_optimizer.apply_gradients(grad_var_gen)

tf.initialize_all_variables().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
iterations = 0

for epoch in range(n_epochs):
    np.random.shuffle(face_images)

    for start, end in zip(
            range(0, len(face_images), batch_size),
            range(batch_size, len(face_images), batch_size)
            ):

        batch_image_files = face_images[start:end]
        batch_images = map(lambda x: crop_resize( os.path.join( face_image_path, x) ), batch_image_files)
        batch_z = np.random.uniform(-1, 1, size=[batch_size, dim_z])

        if np.mod(iterations, 2) == 0:
            _, gen_loss_val, discrim_loss_val = sess.run(
                    [train_op_gen, g_cost_tf, d_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        image_tf:batch_images
                        })
        else:
            _, gen_loss_val, discrim_loss_val = sess.run(
                    [train_op_discrim, g_cost_tf, d_cost_tf],
                    feed_dict={
                        Z_tf:batch_z,
                        image_tf:batch_images
                        })

        iterations += 1

    generated_samples = sess.run(
            image_tf_sample,
            feed_dict={
                Z_tf_sample:Z_np_sample
                })
    generated_samples = (generated_samples + 1.)/2.
    save_visualization(generated_samples, (14,14), save_path='./vis/sample_'+str(epoch)+'.jpg')

