# %% [code]
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

import math
import random
import time
import datetime
import shutil
from tqdm import tqdm, tqdm_notebook

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

import wganlayers
import wganloss

def adjust_lr(opt,lrmax=0.001,lrmin=0.0001):
    # randomly assign a learning rate to both the discriminator and generator each iteration
    newlr = 1./(1./lrmax + tf.random.uniform(shape=(),maxval=1./lrmin))
    opt.lr = newlr

def adjust_lr_prob(opt,lrhigh=0.005,prob=0.003,lrmax=0.001,lrmin=0.0001):#,lrmax=0.001,lrmin=0.0001):
    # randomly assign a learning rate to both the discriminator and generator each iteration
    if tf.random.uniform(shape=(),maxval=1.) < prob:
        opt.lr = lrhigh
    else:
        adjust_lr(opt,lrmax=lrmax,lrmin=lrmin)

## learning rate modifcations
def set_lr(opt, lr):
    opt.lr = lr

# note: not implemented
def adapt_lr_GAN(optg, optd, histg, histd, lrmax=0.001, lrmin=0.00001):
    pauseFade = False
    if len(hist1) > 20:
        rhg = histg[-20:]
        rhd = histd[-20:]
        hdiff = rhg - rhd
        avg = np.mean(rhg)
        avd = np.mean(rhd)
        avdiff = np.mean(hdiff)
        changeg = np.mean(rhg[-10:])-np.mean(rhg[:10])
        changed = np.mean(rhd[-10:])-np.mean(rhd[:10])
        changediff = np.mean(hdiff[-10:])-np.mean(hdiff[:10])
        stdg = np.std(rhg)
        stdd = np.std(rhd)
        stddiff = np.std(hdiff)
        if 3*stdg > stddiff:
            optg.lr = max(optg.lr/2,lrmin)
            optd.lr = max(optd.lr/2,lrmin)
            print("Cond 1 adapt: ".format(optg.lr))
        elif changeg > 5 and avg > 30:
            optg.lr = min(2*optg.lr,lrmax)
            optg.lr = min(2*optg.lr,lrmax)
            print("Cond 2 adapt: ".format(optg.lr))
            pauseFade = True
        elif stdg > changeg and avd < 0:
            optg.lr = max(optg.lr/2,lrmin)
            optd.lr = max(optd.lr/2,lrmin)
            print("Cond 3 adapt: ".format(optg.lr))
        print("avg {}; delg {}; stdg {}".format(avg,changeg, stdg))
        print("avd {}; deld {}; stdd {}".format(avg,changeg, stdg))
        print("avdiff {}; deldiff {}; stddiff {}".format(avg,changeg, stdg))
    return pauseFade

optimizercrit=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=1e-8)
optimizergen=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=1e-8)
