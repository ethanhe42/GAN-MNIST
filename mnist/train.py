import os
import numpy as np
from model import *
from util import *
from load import mnist_with_valid_set

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [28,28,1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1

visualize_dim=196

trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        )

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.global_variables_initializer().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
Y_np_sample = OneHot( np.random.randint(10, size=[visualize_dim]))
iterations = 0
k = 2

step = 200

for epoch in range(n_epochs):
    index = np.arange(len(trY))
    np.random.shuffle(index)
    trX = trX[index]
    trY = trY[index]

    for start, end in zip(
            range(0, len(trY), batch_size),
            range(batch_size, len(trY), batch_size)
            ):

        Xs = trX[start:end].reshape( [-1, 28, 28, 1]) / 255.
        Ys = OneHot(trY[start:end])
        Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        if np.mod( iterations, k ) != 0:
            _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        Y_tf:Ys
                        })
            discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf,p_real,p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
            print("=========== updating G ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        else:
            _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        Y_tf:Ys,
                        image_tf:Xs
                        })
            gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)

        print("Average P(real)=", p_real_val.mean())
        print("Average P(gen)=", p_gen_val.mean())

        if np.mod(iterations, step) == 0:
            generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample:Z_np_sample,
                        Y_tf_sample:Y_np_sample
                        })
            generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (14,14), save_path='./vis/sample_%04d.jpg' % int(iterations/step))

        iterations += 1

