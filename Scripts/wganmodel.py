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


# model save and load functions
def saveh5(model, name):
    filename = h5DIR + name + '.h5'
    model.save(filename)

def saveh5s(model, name):
    saveh5(model.generator, 'gen_'+ name)
    saveh5(model.discriminator, 'crit_'+ name)

customOs={'LeakyReLU': LeakyReLU,'Conv2DELR': Conv2DELR,'PixelNorm':PixelNorm,
    'AddWithFade':AddWithFade,'MBStDev':MBStDev}
def loadh5s(name):
    filename = h5inDIR + 'gen_' +name + '.h5'
    generator = load_model(filename,custom_objects=customOs)
    filename = h5inDIR + 'crit_' +name + '.h5'
    critic = load_model(filename,custom_objects=customOs)
    return generator, critic

# implementation of wasserstein loss with gradient penalty
# Useful information: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
# keras implementation: https://keras.io/examples/generative/wgan_gp/
class WGANGP(keras.Model):
    def __init__(
                 self,
                 discriminator,
                 generator,
                 latent_dim,
                 discriminator_steps=1,
                 gp_weight=10.0,
                 #nMaxFade = 800000 this is the recommended value for faces
                 nMaxFade = 100000
                 ):
        super(WGANGP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_steps
        self.gp_weight = gp_weight
        self.nMaxFade = nMaxFade
        self.fade = 0.0
        self.nRunFade = 0
        self.trainFreeze = False
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    
    def get_config(self):
        base_config = super(keras.Model,self).get_config()
        base_config["discriminator"] = self.discriminator
        base_config["d_optimizer"] = self.d_optimizer
        base_config["d_loss_fn"] = self.d_loss_fn
        base_config["generator"] = self.generator
        base_config["g_loss_fn"] = self.g_loss_fn
        base_config["g_optimizer"] = self.g_optimizer
        base_config["latent_dim"] = self.latent_dim
        base_config["discriminator_steps"] = self.d_steps
        base_config["gp_weight"] = self.gp_weight
        base_config["nMaxFade"] = self.nMaxFade
        return base_config
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
            
            This loss is calculated on an interpolated image
            and added to the discriminator loss.
            """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)
        
        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images, withFade = True, adaptlr = False, randg=False, randd=False):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        
        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        
        #print("Batch Size {}".format(batch_size))
        
        # update the fade is model is being run with fade
        if self.fade < 1:
            if withFade and not self.trainFreeze:
                self.nRunFade += batch_size
                self.fade = min(self.nRunFade,self.nMaxFade)/self.nMaxFade
                update_fade(self.generator,self.fade)
                update_fade(self.discriminator,self.fade)
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.
        
        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_images, training=True)
                
                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

                if randd:
                    adjust_lr(self.d_optimizer)
                
                # Get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                
                self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
                
                # Train the generator now.
                # Get the latent vector
                random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                with tf.GradientTape() as tape:
                    # Generate fake images using the generator
                    generated_images = self.generator(random_latent_vectors, training=True)
                    # Get the discriminator logits for fake images
                    gen_img_logits = self.discriminator(generated_images, training=True)
                    # Calculate the generator loss
                    g_loss = self.g_loss_fn(gen_img_logits)
                    
                    if randg:
                        adjust_lr_prob(self.g_optimizer)
                    # Get the gradients w.r.t the generator loss
                    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
                    # Update the weights of the generator using the generator optimizer
                      
                    self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}


    
# toggle trainability of WGAN
def toggle_train(WGAN, trainTog=True, size=128):
    if trainTog == False:
        WGAN.trainFreeze = True
        for layer in WGAN.discriminator.layers:
            if layer.output.shape[1] != size:
                layer.trainable = False
        for layer in WGAN.generator.layers:
            if layer.output.shape[1] != size:
                layer.trainable = False
    else:
        WGAN.trainFreeze = False
        for layer in WGAN.discriminator.layers:
            layer.trainable = True
        for layer in WGAN.generator.layers:
            layer.trainable = True     
            
def saveWGAN(model, name):
    filename = h5DIR + 'gen_'+ name 
    model.generator.save(filename)
    filename = h5DIR + 'crit_'+ name 
    model.discriminator.save(filename)
    filename = h5DIR + 'weights_'+ name
    model.save_weights(filename)

# dictionary for the number of filters per layer
leveldict = {4: {'filters': 256},
            8: {'filters': 256},
            16: {'filters': 256},
            32: {'filters': 256},
            64: {'filters': 128},
            128: {'filters': 64}}

# take from a model that criticises an nxnx3 figure to one that criticizes a 2nx2nx3 figure
def add_critic_level(old_c_model, new_level=8, nSkip=3,nSkipPassby=1,nColors=3,nameX=None):
    namebase = nameX
    if nameX == None:
        namebase = str(new_level)
        
    thisleveldict = leveldict[new_level]
    filters = thisleveldict['filters']
    conv2filters = leveldict[new_level/2]['filters']
    
    #define new input layer
    layer_in = Input(shape=(new_level, new_level, nColors,),name="Input_C0_{}".format(namebase)) 
    
    # new round start passby here
    conv_0 = Conv2DELR(filters, (1,1), padding="SAME",name="Conv_C0_{}".format(namebase))(layer_in)
    act_0 = LeakyReLU(alpha=0.2,name="Act_C0_{}".format(namebase))(conv_0)
    # first layer set (newround start main model here)
    conv_1 = Conv2DELR(filters, (3,3), padding="SAME",name="Conv_C1_{}".format(namebase))(act_0)
    act_1 = LeakyReLU(alpha=0.2,name="Act_C1_{}".format(namebase))(conv_1)
    # second layer set
    conv_2 = Conv2DELR(conv2filters, (3,3), padding="SAME",name="Conv_C2_{}".format(namebase))(act_1)
    act_2 = LeakyReLU(alpha=0.2,name="Act_C2_{}".format(namebase))(conv_2)
    newcriticlayers = AveragePooling2D(name="AvPool_C0_{}".format(namebase))(act_2)
    critic = newcriticlayers
    # append the earlier model
    for i in range(nSkip,len(old_c_model.layers)):
        critic = old_c_model.layers[i](critic)
    criticModel = Model(layer_in, critic)
    
    # define passby model, first downsample
    pathtofade = AveragePooling2D(name="AvPool_CPass_{}".format(namebase))(layer_in)
    # note: including conv_0, and act_0 from previous layer
    for i in range(nSkipPassby,nSkip):
        pathtofade = old_c_model.layers[i](pathtofade)
    critic_fade = AddWithFade(name="Fade_CPass_{}".format(namebase))([pathtofade,newcriticlayers])
    for i in range(nSkip,len(old_c_model.layers)):
        critic_fade = old_c_model.layers[i](critic_fade)
    criticFadeOut = Model(layer_in, critic_fade)
    return [criticModel, criticFadeOut]

# take from a model that generates an nxnx3 figure to one that generates a 2nx2nx3 figure
def add_generator_level(old_g_model, new_level=8, nDrop=3,nDropPassby=1,nColors=3,nameX=None):
    namebase = nameX
    if nameX == None:
        namebase = str(new_level)
        
    thisleveldict = leveldict[new_level]
    filters = thisleveldict['filters']
    
    # get input
    layer_in = old_g_model.input
    # remove final layer
    newendofold = old_g_model.layers[-2].output 
    sizeaugment = UpSampling2D(name="UpSample_G0_{}".format(namebase))(newendofold)
    
    # first layer set
    conv_1 = Conv2DELR(filters, (3,3), padding="SAME",name="Conv_G1_{}".format(namebase))(sizeaugment)
    pixnorm_1 = PixelNorm(name="Pix_G1_{}".format(namebase))(conv_1)
    act_1 = LeakyReLU(alpha=0.2,name="Act_G1_{}".format(namebase))(pixnorm_1)
    # second layer set
    conv_2 = Conv2DELR(filters, (3,3), padding="SAME",name="Conv_G2_{}".format(namebase))(act_1)
    pixnorm_2 = PixelNorm(name="Pix_G2_{}".format(namebase))(conv_2)
    act_2 = LeakyReLU(alpha=0.2,name="Act_G2_{}".format(namebase))(pixnorm_2)
    
    # save for combo below
    conv_out = Conv2DELR(nColors, (1,1), padding="SAME",name="Conv_Gout_{}".format(namebase))(act_2)
    genModel = Model(layer_in, conv_out)
    
    # define passby model
    #old_end = old_g_model.layers[-1].output
    #sizeaugment_fade = UpSampling2D(name="UpSample_GPass_{}".format(namebase))(old_end)
    old_end = old_g_model.layers[-1]
    sizeaugment_fade = old_end(sizeaugment)
    conv_out_fade = AddWithFade(name="Fade_GPass_{}".format(namebase))([sizeaugment_fade,conv_out])
    
    genModelFade = Model(layer_in, conv_out_fade)
    
    return [genModel, genModelFade]

# nLevels = number of pixelation levels (4,8,16,32,64,128) 
def create_critics(nColors=3, initialSize = 4, nLevels = 6):
    # first create the lowest level critic
    thisleveldict = leveldict[initialSize]
    filters = thisleveldict['filters']
    
    #define new input layer
    layer_in = Input(shape=(initialSize, initialSize, nColors,)) 
    # new round start passby here
    # define new input processing layer
    conv_0 = Conv2DELR(filters, (1,1), padding="SAME")(layer_in)
    act_0 = LeakyReLU(alpha=0.2)(conv_0)
    # first layer set (newround start main model here)
    # apply minibatch standard deviation
    miniBDev = MBStDev()(act_0)
    conv_1 = Conv2DELR(filters, (3,3), padding="SAME")(miniBDev)
    act_1 = LeakyReLU(alpha=0.2)(conv_1)
    # second layer set (the end is a 4x4 convolution)
    conv_2 = Conv2DELR(filters, (4,4), padding="SAME")(act_1)
    act_2 = LeakyReLU(alpha=0.2)(conv_2)
    dense_out = Flatten()(act_2)
    out_classifier = Dense(1)(dense_out)
    
    # define and compile model
    initial_critic = Model(layer_in,out_classifier)
    
    # collect all models
    modellist = [[initial_critic,initial_critic]]
    curlevel = 4
    for i in range(1,nLevels):
        curlevel = curlevel * 2
        # modellist[-1][0] corresponds to the version with no fade
        newmodels = add_critic_level(modellist[-1][0], new_level=curlevel)
        modellist.append(newmodels)
    return modellist

# nLevels = number of pixelation levels (4,8,16,32,64,128) 
def create_gens(nInputs =128, nColors=3, initialSize = 4, nLevels = 6):
    # first create the lowest level critic
    thisleveldict = leveldict[initialSize]
    filters = thisleveldict['filters']
    # #4.1 weight initialization and maxnorm constraint
    kinit = tf.keras.initializers.RandomNormal(stddev=1.)
    
    #define new input layer
    layer_in = Input(shape=(nInputs,))
    dense_0 = Dense(nInputs*initialSize*initialSize, kernel_initializer=kinit)(layer_in)
    reshape_0 = Reshape((initialSize, initialSize, nInputs))(dense_0)
    #may want to add activiation functions
    # first layer set (start with 4x4)
    conv_1 = Conv2DELR(filters, (4,4), padding="SAME")(reshape_0)
    pixnorm_1 = PixelNorm()(conv_1)
    act_1 = LeakyReLU(alpha=0.2)(pixnorm_1)
    # second layer set 
    conv_2 = Conv2DELR(filters, (3,3), padding="SAME")(act_1)
    pixnorm_2 = PixelNorm()(conv_2)
    act_2 = LeakyReLU(alpha=0.2)(pixnorm_2)
    # save for combo below
    conv_out = Conv2DELR(nColors, (1,1), padding="SAME")(act_2)
    genModel = Model(layer_in, conv_out)
    
        
    # collect all models
    modellist = [[genModel,genModel]]
    curlevel = 4
    for i in range(1,nLevels):
        curlevel = curlevel * 2
        # modellist[-1][0] corresponds to the version with no fade
        newmodels = add_generator_level(modellist[-1][0], new_level=curlevel)
        modellist.append(newmodels)
    return modellist

def create_gans(gens,crits,nInputs = 128):
    ganlist = []
    assert len(gens)==len(crits), "Generators and Discriminators created with different lengths"
    for i in range(len(gens)):
        with strategy.scope():
            # compile standard
            wgan1 = WGANGP(discriminator=crits[i][0],generator=gens[i][0],latent_dim=nInputs,discriminator_steps=1)
            wgan1.compile(d_optimizer=optimizercrit,g_optimizer=optimizergen,g_loss_fn=generator_loss,d_loss_fn=critic_loss)
            # compile fade
            wgan2 = WGANGP(discriminator=crits[i][1],generator=gens[i][1],latent_dim=nInputs,discriminator_steps=1)
            wgan2.compile(d_optimizer=optimizercrit,g_optimizer=optimizergen,g_loss_fn=generator_loss,d_loss_fn=critic_loss)
            # add to gan list
            ganlist.append([wgan1,wgan2])
    return ganlist
