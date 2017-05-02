# Denoising with GAN
[Paper](https://uofi.box.com/shared/static/s16nc93x8j6ctd0ercx9juf5mqmqx4bp.pdf) | [Video](https://www.youtube.com/watch?v=Yh_Bsoe-Qj4)

## Introduction

Animation movie companies like Pixar and Dreamworks render their 3d scenes using a technique called Pathtracing which enables them to create high quality photorealistic frames. Pathtracing involves shooting 1000’s of rays into a pixel randomly(Monte Carlo) which will then hit the objects in the scene and based on the reflective property of the object the rays reflect or refract or get absorbed. The colors returned by these rays are averaged to get the color of the pixel and this process is repeated for all the pixels. Due to the computational complexity it might take 8-16 hours to render a single frame. 

We are proposing a neural network based solution for reducing 8-16 hours to a couple of seconds using a Generative Adversarial Network. The main idea behind this proposed method is to render using small number of samples per pixel (let say 4 spp or 8 spp instead of 32K spp) and pass the noisy image to our network, which will generate a photorealistic image with high quality. 

# Demo Video [Link](https://www.youtube.com/watch?v=Yh_Bsoe-Qj4)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Yh_Bsoe-Qj4/0.jpg)](https://www.youtube.com/watch?v=Yh_Bsoe-Qj4)



#### Table of Contents

* [Installation](#installation)
* [Running](#running)
* [Dataset](#dataset)
* [Hyperparameters](#hyperparameter)
* [Results](#results)
* [Improvements](#improvements)
* [Credits](#credits)

## Installation

To run the project you will need:
 * python 3.5
 * tensorflow (v1.1 or v1.0)
 * PIL
 * [CKPT FILE](https://uofi.box.com/shared/static/21a5jwdiqpnx24c50cyolwzwycnr3fwe.gz)
 * [Dataset](https://uofi.box.com/shared/static/gy0t3vgwtlk1933xbtz1zvhlakkdac3n.zip)

## Running

Once you have all the depenedencies ready, do the folowing:

Download the dataset extract it to a folder named 'dataset' (ONLY if you want to train, not needed to run).

Extract the CKPT files to a folder named 'Checkpoints'

Run main.py -- python3 main.py

Go to the browser, if you are running it on a server then [ip-address]:8888, if you are on your local machine then localhost:8888

## Dataset
We picked random 40 images from pixar movies, added gaussian noise of different standard deviation, 5 sets of 5 different standard deviation making a total of 1000 images for the training set. For validation we used 10 images completely different from the training set and added gaussian noise. For testing we had both added gaussian images and real noisy images.

## Hyperparameters
* Number of iterations - 10K
* Adversarial Loss Factor - 0.5
* Pixel Loss Factor - 1.0
* Feature Loss Factor - 1.0
* Smoothness Loss Factor - 0.0001

## Results
3D rendering test data:
<img src="https://github.com/manumathewthomas/CS523Project3/blob/master/result1.PNG" alt="alt text" width="960" height="480">

Real noise images:
<img src="https://github.com/manumathewthomas/CS523Project3/blob/master/result2.png" alt="alt text" width="960" height="480">

CT-Scan:
<img src="https://github.com/manumathewthomas/CS523Project3/blob/master/result3.PNG" alt="alt text" width="960" height="480">
 

## Improvements

* Increase the num of iteration to 100K.
* Train the network for different noises.
* Make it work on a real-time app.

## Credits

* [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)
* [Image De-raining using conditional generative adversarial network](https://arxiv.org/pdf/1701.05957.pdf)
* [Creating photorealistic images from gameboy camera](http://www.pinchofintelligence.com/photorealistic-neural-network-gameboy/)
* [CS20SI](cs20si.stanford.edu)
* [CS231n](https://cs231n.github.io/)
