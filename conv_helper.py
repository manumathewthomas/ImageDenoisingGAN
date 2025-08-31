import tensorflow as tf
from utils import *

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, ksize, stride, name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(out_channels, ksize, strides=stride, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(out_channels, ksize, strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.2)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return self.add([x, inputs])

class DeconvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, name=None):
        super(DeconvolutionBlock, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs, target_height, target_width):
        x = tf.image.resize(inputs, (target_height, target_width), method='bilinear')
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)
