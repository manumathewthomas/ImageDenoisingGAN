import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import *

def conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False):
    with tf.variable_scope(scope_name):
        filter = tf.Variable(tf.random_normal([ksize, ksize, in_channels, out_channels], stddev=0.03))
        output = tf.nn.conv2d(input_image, filter, strides=[1, stride, stride, 1], padding='SAME')
        output = slim.batch_norm(output)
        if activation_function:
            output = activation_function(output)
        return output, filter

def residual_layer(input_image, ksize, in_channels, out_channels, stride, scope_name):
    with tf.variable_scope(scope_name):
        output, filter = conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name+"_conv1")
        output, filter = conv_layer(output, ksize, out_channels, out_channels, stride, scope_name+"_conv2")
        output = tf.add(output, tf.identity(input_image))
        return output, filter

def transpose_deconvolution_layer(input_tensor, used_weights, new_shape, stride, scope_name):
    with tf.varaible_scope(scope_name):
        output = tf.nn.conv2d_transpose(input_tensor, used_weights, output_shape=new_shape, strides=[1, stride, stride, 1], padding='SAME')
        output = tf.nn.relu(output)
        return output

def resize_deconvolution_layer(input_tensor, new_shape, scope_name):
    with tf.variable_scope(scope_name):
        output = tf.image.resize_images(input_tensor, (new_shape[1], new_shape[2]), method=1)
        output, unused_weights = conv_layer(output, 3, new_shape[3]*2, new_shape[3], 1, scope_name+"_deconv")
        return output

def deconvolution_layer(input_tensor, new_shape, scope_name):
    return resize_deconvolution_layer(input_tensor, new_shape, scope_name)

def output_between_zero_and_one(output):
    output +=1
    return output/2
