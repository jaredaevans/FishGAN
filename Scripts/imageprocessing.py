# %% [code]
import os
import sys

import tensorflow as tf

#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

#for GIF
import glob
import imageio

import random

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from IPython.display import Image as IpyImage

#note: poor structuring has these hardcoded
FishDIR='/kaggle/input/cleanedfish/'
GIFDIR = '/kaggle/working/GIFs/'

# import / exporting
def read_image(src):
    img = cv2.imread(src)
    if img is None:
        print(src)
        raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_image(img,filename):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

# GIFs
#IpyImage(filename=GIFDIR+modelname+"_test.gif")
def constructGIF(ix,filename):
    gif_path = GIFDIR+filename+".gif"
    frames_path = GIFDIR+filename+"_{i}.jpg"
    with imageio.get_writer(gif_path, mode='I') as writer:
        for i in range(ix):
            writer.append_data(imageio.imread(frames_path.format(i=i)))

def write_images_and_GIF(imgsetforGIF,modelname):
    frames_path = GIFDIR+modelname+"_{i}.jpg"
    for i, imgs in enumerate(imgsetforGIF):
        plot_multiple_images(imgs, 8)
        plt.savefig(frames_path.format(i=i))
        plt.close()
    constructGIF(len(imgsetforGIF),modelname)

# plotting functions
def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    plt.subplots_adjust(hspace=0.03, wspace=0)
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1,snap=True)
        plt.imshow(np.clip((image + 1.)/2.,0.,1.), cmap="binary")
        #plt.imshow(image, cmap="binary")
        plt.axis("off")

# History plot
def plot_history(d_hist, g_hist):
    plt.plot(d_hist, label='crit')
    plt.plot(g_hist, label='gen')
    plt.legend()
    plt.show()
    plt.savefig(GIFDIR+'/plot_line_plot_loss.png')
    plt.close()

# Image preprocessing
datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.05,
        height_shift_range=0.1,
        shear_range=10,
        zoom_range=[0.95,1.2],
        brightness_range=[0.8,1.4],
        horizontal_flip=True,
        fill_mode='nearest')

# adjust the saturation of the image
def colorSaturationAdjust(img, satRange=[0.6,1.4]):
    rSat = random.triangular(satRange[0],satRange[1])
    gSat = random.triangular(satRange[0],satRange[1])
    bSat = random.triangular(satRange[0],satRange[1])
    satVar = [rSat, gSat, bSat]
    return np.clip(img * satVar,0,1)

# run IDG and color adjuster on a list of images
def multiprocessingI(imglist,new_length,batch_size):
    imglistproc = []
    for img in datagen.flow((imglist+1.)/2.000,batch_size=batch_size)[0]:
        newimg = np.clip(2.*(colorSaturationAdjust(img/255.0)-0.5),-1.,1.)
        imglistproc.append(cv2.resize(newimg, (new_length,new_length), interpolation = cv2.INTER_AREA))
        #imglistproc.append(colorSaturationAdjust(img/255.0))
    return np.asarray(imglistproc)

