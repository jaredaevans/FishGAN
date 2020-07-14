# FishGAN
Construction of a fish generator  
 
Why fish?  Many fish are very beautiful with extremes in colorations, and shapes, and patterns that have a tremendous spread in presentation.  Rather than aspiring to construct a generator that would make a fish that looks obviously like a subtly different version of an existing fish, my hope is to construct a generator that can make beautiful images of fish that look both unlike any fish I have seen, yet clearly fishy.  

## WGAN-GP (current version)
![WGAN-GP GIF](Notebooks/WGAN-GP/WGAN-GP_32.gif)   
This is heavily influenced by ProgressiveGAN (below), but I did not use the progressive aspect. Overall, a big success.  I shrunk the figures to 32x32 and then used those to train the GAN.  See Notebooks/WGAN-GP for specific implementation.

## ProgressiveGAN with WGAN-GP
- very promising, but very slow to train
- could get nice images up to 16x16, were very unstable at 32x32, and dissolved at 64x64
- notebook FishWGAN is uploaded

## Simple DCGAN 
- First attempt at a GAN, went okay - mostly, all modifications made it worse
