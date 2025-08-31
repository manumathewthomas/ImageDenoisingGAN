import time
import tensorflow as tf
import numpy as np
import cv2
from utils import *
from model import generator, discriminator
from skimage import measure

def test(image):
    tf.keras.backend.clear_session()

    # Process image
    image = tf.image.per_image_standardization(image)
    image = generator.predict(image)
    image = np.resize(image[0][56:, :, :], [144, 256, 3])
    imsave('output', image)
    return image

def denoise(image):
    # Handle both file paths and numpy arrays
    if isinstance(image, str):
        # If image is a file path
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image.astype('float32')
    elif isinstance(image, np.ndarray):
        # If image is already a numpy array, ensure it's in the right format
        if image.dtype != np.float32:
            image = image.astype('float32')
        if image.max() > 1.0:
            image = image / 255.0
        # Ensure 3 channels (RGB)
        if image.shape[-1] == 4:  # If RGBA
            image = image[..., :3]  # Take only RGB channels
    else:
        raise ValueError("Input must be either a file path (str) or a numpy array")

    npad = ((56, 56), (0, 0), (0, 0))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    output = test(image)
    return output

if __name__=='__main__':
    image = cv2.imread(sys.argv[-1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.astype('float32') / 255.0

    npad = ((56, 56), (0, 0), (0, 0))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    test(image)
