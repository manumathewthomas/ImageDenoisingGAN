# Denoising with GAN

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

Download the dataset friends.txt and move it to data/lightweight/. 

Extract the ckpt zip file, you will get a folder 'model-server' with the ckpt file and its associated files. Move this folder to /save. The server will look at the model present on save/model-server/model.ckpt

To configure the web app

```bash
export CHATBOT_SECRET_KEY="my-secret-key"
cd chatbot_website/
python manage.py makemigrations
python manage.py migrate
```

Then, to launch the server locally, use the following commands:

```bash
cd chatbot_website/
redis-server &  # Launch Redis in background
python manage.py runserver 0.0.0.0:8888
```
Go to the browser, if you are running it on a server then [ip-address]:8888, if you are on your local machine then localhost:8888

## Dataset
For this project we used the [Friends TV Corpus](https://sites.google.com/site/friendstvcorpus/), which was currated by David Ayliffe for his masters thesis to study inter-gender and intra-gender conversation. We formatted the data to get conversation between Joey(a character) and several other characters. There are two reasons for this: 
  
  * We were targeting sentences with 5 words. 

  * Joey has the highest number of conversations in the dataset.

## Hyperparameters
Some of the hyperparameter we played with are sentence-length, number of hidden-layers, word embedding-size, batch-size and number of iterations. One thing to note here is that there is no logic behind these values, it was purely based on a trial and error approach.

  * We started training with sentences containing more than 20 words, It took more than 16 hours to train and still it was giving gibberish output. Then we reduced it to 10 words, although it was comparitively better we were not happy with the result. Finally we tried with 5 words and we got good outputs.
  
  * There are 2 RNN layers(encoder and decoder) and for each layer we put the number of hidden layers to 2048.
  * Changed the word embedding size to 256.
  * Batch size to 512.
  * Number of iteration to 200,000.

## Results

Below are some screenshots of our chat with the chatbot. It gives preety good results for standard questions as well as some character specific questions. 

Good results
* The bot replies with the character name when asked : Who are you?
* Has a flirtatious nature just like Joey! 
* Understands various greetings like Hi, Hello, Hey and responds accurately. 
* Is smart enough to respond with a valid number when questioned: What time is it?

<img src="https://github.com/manumathewthomas/Chat-with-Joey/blob/master/Results/Good.png" alt="alt text" width="850" height="500">

Random 

It fails to respond appropriately to some questions.
Q: What's up?
A: Kidney Stones!

<img src="https://github.com/manumathewthomas/Chat-with-Joey/blob/master/Results/Random1.png" alt="alt text" width="850" height="500">

<img src="https://github.com/manumathewthomas/Chat-with-Joey/blob/master/Results/Random2.png" alt="alt text" width="850" height="500">




 

## Improvements

* Increase the sentence length from 5 to 10.
* Train multple characters from the show.
* Make bots chat with each other.

## Credits

* [A Neural Conversational Model](http://arxiv.org/abs/1506.05869)
* [DeepQA repo](https://github.com/Conchylicultor/DeepQA)
* [seq2seq model RNN](https://www.tensorflow.org/tutorials/seq2seq)
