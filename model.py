import numpy as np
import tensorflow as tf
from utils import *
from conv_helper import ResidualBlock, DeconvolutionBlock

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 9, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(3, 9, padding='same', activation='tanh')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.activation = tf.keras.layers.LeakyReLU(0.2)
        
        # Create residual blocks
        self.res_blocks = [ResidualBlock(128, 3, 1, name=f"g_res_{i}") for i in range(3)]
        
        # Create deconvolution blocks
        self.deconv1 = DeconvolutionBlock(64, name='g_deconv1')
        self.deconv2 = DeconvolutionBlock(32, name='g_deconv2')

        # Add layer for final normalization
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def normalize_output(self, x):
        return (x + 1.0) / 2.0

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        conv1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Deconvolution layers
        x = self.deconv1(x, target_height=128, target_width=128)
        x = self.deconv2(x, target_height=256, target_width=256)
        x = self.add1([x, conv1])

        x = self.conv4(x)
        x = self.add2([x, inputs])
        return self.normalize_output(x)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(48, 4, strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(96, 4, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(192, 4, strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(384, 4, padding='same')
        self.conv5 = tf.keras.layers.Conv2D(1, 4, padding='same', activation='sigmoid')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        
        self.activation = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        return self.conv5(x)

# Create model instances
generator = Generator()
discriminator = Discriminator()

# Build the models with a sample input to initialize weights
dummy_input = tf.zeros((1, 256, 256, 3))
_ = generator(dummy_input)
_ = discriminator(dummy_input)
