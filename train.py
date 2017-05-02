import time

import tensorflow as tf
import numpy as np

from utils import *
from model import *

from skimage import measure



def train():
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')

    Gz = generator(gen_in)
    Dx = discriminator(real_in)
    Dg = discriminator(Gz, reuse=True)

    real_in_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), real_in)
    Gz_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), Gz)

    psnr=0
    ssim=0

    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg))
    g_loss = ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.log(Dg)) + PIXEL_LOSS_FACTOR * get_pixel_loss(real_in, Gz) \
             + STYLE_LOSS_FACTOR * get_style_loss(real_in_bgr, Gz_bgr) + SMOOTH_LOSS_FACTOR * get_smooth_loss(Gz)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_solver = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_solver = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        start_time = time.time()
        n_batches = 200
        total_iteration = n_batches * N_EPOCHS

        validation_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), validation))


        for index in range(initial_step, total_iteration):
            input_batch = load_next_training_batch()
            training_batch, groundtruth_batch = np.split(input_batch, 2, axis=2)

            training_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), training_batch))
            groundtruth_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), groundtruth_batch))


            _, d_loss_cur = sess.run([d_solver, d_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})
            _, g_loss_cur = sess.run([g_solver, g_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})




            if(index + 1) % SKIP_STEP == 0:

                saver.save(sess, CKPT_DIR, index)
                image = sess.run(Gz, feed_dict={gen_in: validation_batch})
                image = np.resize(image[7][56:, :, :], [144, 256, 3])

                imsave('val_%d' % (index+1), image)
                image = scipy.misc.imread(IMG_DIR+'val_%d.png' % (index+1), mode='RGB').astype('float32')
                psnr = measure.compare_psnr(metrics_image, image, data_range=255)
                ssim = measure.compare_ssim(metrics_image, image, multichannel=True, data_range=255, win_size=11)

                print(
                    "Step {}/{} Gen Loss: ".format(index + 1, total_iteration) + str(g_loss_cur) + " Disc Loss: " + str(
                        d_loss_cur)+ " PSNR: "+str(psnr)+" SSIM: "+str(ssim))



if __name__=='__main__':
    training_dir_list = training_dataset_init()
    validation = load_validation()
    train()
