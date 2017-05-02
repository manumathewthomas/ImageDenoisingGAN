import os
import re
import sys
import glob
import scipy.misc
from itertools import cycle

import numpy as np
import tensorflow as tf


from libs import vgg16

from PIL import Image


LEARNING_RATE = 0.002
BATCH_SIZE = 5
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]
SKIP_STEP = 10
N_EPOCHS = 500
CKPT_DIR = './Checkpoints/'
IMG_DIR = './Images/'
GRAPH_DIR = './Graphs/'
TRAINING_SET_DIR= './dataset/training/'
# GROUNDTRUTH_SET_DIR='./dataset/groundtruth/'
VALIDATION_SET_DIR='./dataset/validation/'
METRICS_SET_DIR='./dataset/metrics/'
TRAINING_DIR_LIST = []
ADVERSARIAL_LOSS_FACTOR = 0.5
PIXEL_LOSS_FACTOR = 1.0
STYLE_LOSS_FACTOR = 1.0
SMOOTH_LOSS_FACTOR = 1.0
metrics_image = scipy.misc.imread(METRICS_SET_DIR+'gt.png', mode='RGB').astype('float32')


def initialize(sess):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver

def get_training_dir_list():
    training_list = [d[1] for d in os.walk(TRAINING_SET_DIR)]
    global TRAINING_DIR_LIST
    TRAINING_DIR_LIST = training_list[0]
    return TRAINING_DIR_LIST

def load_next_training_batch():
    batch = next(pool)

    # filelist = sorted(glob.glob(TRAINING_SET_DIR+ batch +'/*.png'), key=alphanum_key)
    # batch = np.array([np.array(scipy.misc.imread(fname, mode='RGB').astype('float32')) for fname in filelist])
    # npad =((0, 0), (56, 56), (0, 0), (0, 0))
    # batch = np.pad(batch, pad_width=npad, mode='constant', constant_values=0)
    return batch

# def load_groundtruth():
#     filelist = sorted(glob.glob(GROUNDTRUTH_SET_DIR + '/*.png'), key=alphanum_key)
#     groundtruth = np.array([np.array(scipy.misc.imread(fname, mode='RGB').astype('float32')) for fname in filelist])
#     # npad = ((0, 0), (56, 56), (0, 0), (0, 0))
#     # groundtruth = np.pad(groundtruth, pad_width=npad, mode='constant', constant_values=0)
#     return groundtruth

def load_validation():
    filelist = sorted(glob.glob(VALIDATION_SET_DIR + '/*.png'), key=alphanum_key)
    validation = np.array([np.array(scipy.misc.imread(fname, mode='RGB').astype('float32')) for fname in filelist])
    npad = ((0, 0), (56, 56), (0, 0), (0, 0))
    validation = np.pad(validation, pad_width=npad, mode='constant', constant_values=0)
    return validation

def training_dataset_init():
    filelist = sorted(glob.glob(TRAINING_SET_DIR + '/*.png'), key=alphanum_key)
    batch = np.array([np.array(scipy.misc.imread(fname, mode='RGB').astype('float32')) for fname in filelist])
    batch = split(batch, BATCH_SIZE)
    training_dir_list = get_training_dir_list()
    global pool
    pool = cycle(batch)
    # return training_dir_list


def imsave(filename, image):
    scipy.misc.imsave(IMG_DIR+filename+'.png', image)

def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.fromarray(np.uint8(file1))
    image2 = Image.fromarray(np.uint8(file2))

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def RGB_TO_BGR(img):
    img_channel_swap = img[..., ::-1]
    # img_channel_swap_1 = tf.reverse(img, axis=[-1])
    return img_channel_swap


def get_pixel_loss(target,prediction):
    pixel_difference = target - prediction
    pixel_loss = tf.nn.l2_loss(pixel_difference)
    return pixel_loss

def get_style_layer_vgg16(image):
    net = vgg16.get_vgg_model()
    style_layer = 'conv2_2/conv2_2:0'
    feature_transformed_image = tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': image},return_elements=[style_layer])
    feature_transformed_image = (feature_transformed_image[0])
    return feature_transformed_image

def get_style_loss(target,prediction):
    feature_transformed_target = get_style_layer_vgg16(target)
    feature_transformed_prediction = get_style_layer_vgg16(prediction)
    feature_count = tf.shape(feature_transformed_target)[3]
    style_loss = tf.reduce_sum(tf.square(feature_transformed_target-feature_transformed_prediction))
    style_loss = style_loss/tf.cast(feature_count, tf.float32)
    return style_loss

def get_smooth_loss(image):
    batch_count = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    horizontal_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height, image_width-1,3])
    horizontal_one_right = tf.slice(image, [0, 0, 1,0], [batch_count, image_height, image_width-1,3])
    vertical_normal = tf.slice(image, [0, 0, 0,0], [batch_count, image_height-1, image_width,3])
    vertical_one_right = tf.slice(image, [0, 1, 0,0], [batch_count, image_height-1, image_width,3])
    smooth_loss = tf.nn.l2_loss(horizontal_normal-horizontal_one_right)+tf.nn.l2_loss(vertical_normal-vertical_one_right)
    return smooth_loss

