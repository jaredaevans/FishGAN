# %% [code]
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

import math

# from https://keras.io/examples/generative/wgan_gp/
# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def critic_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    #
    # this is a simple implementation of the drift loss
    epsilonDrift = 0.001
    epsilonLoss = epsilonDrift * tf.reduce_mean(tf.nn.l2_loss(real_img))
    return fake_loss - real_loss + epsilonLoss

# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)
