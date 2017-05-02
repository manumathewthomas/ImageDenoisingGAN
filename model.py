import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


from utils import *
from conv_helper import *


def generator(input):
    conv1, conv1_weights = conv_layer(input, 9, 3, 32, 1, "g_conv1")
    conv2, conv2_weights = conv_layer(conv1, 3, 32, 64, 1, "g_conv2")
    conv3, conv3_weights = conv_layer(conv2, 3, 64, 128, 1, "g_conv3")

    res1, res1_weights = residual_layer(conv3, 3, 128, 128, 1, "g_res1")
    res2, res2_weights = residual_layer(res1, 3, 128, 128, 1, "g_res2")
    res3, res3_weights = residual_layer(res2, 3, 128, 128, 1, "g_res3")

    deconv1 = deconvolution_layer(res3, [BATCH_SIZE, 128, 128, 64], 'g_deconv1')
    deconv2 = deconvolution_layer(deconv1, [BATCH_SIZE, 256, 256, 32], "g_deconv2")

    deconv2 = deconv2 + conv1

    conv4, conv4_weights = conv_layer(deconv2, 9, 32, 3, 1, "g_conv5", activation_function=tf.nn.tanh)

    conv4 = conv4 + input
    output = output_between_zero_and_one(conv4)

    return output

def discriminator(input, reuse=False):
    conv1, conv1_weights = conv_layer(input, 4, 3, 48, 2, "d_conv1", reuse=reuse)
    conv2, conv2_weights = conv_layer(conv1, 4, 48, 96, 2, "d_conv2", reuse=reuse)
    conv3, conv3_weights = conv_layer(conv2, 4, 96, 192, 2, "d_conv3", reuse=reuse)
    conv4, conv4_weights = conv_layer(conv3, 4, 192, 384, 1, "d_conv4", reuse=reuse)
    conv5, conv5_weights = conv_layer(conv4, 4, 384, 1, 1, "d_conv5", activation_function=tf.nn.sigmoid, reuse=reuse)

    return conv5
