# %% [code]
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K

# %% [code]
# turns all arguments to to float32
def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

# from 1710.10196 use mini batch standard dev on discriminator output (only needs to be done once)
# need to combine layers to fade out, inherit Add class
# fade parameter is updated in WGANGP class
class AddWithFade(Add):
    def __init__(self, fade=0.0, **kwargs):
        super(AddWithFade, self).__init__(**kwargs)
        # fade will increase linearly from 0-1 
        self.fade = K.variable(fade, name='fade_param')
 
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        #input[0] = lower res layer, input[1] = higher res layer 
        output = ((1. - self.fade) * inputs[0]) + (self.fade * inputs[1])
        return output
    
    def get_config(self):
        base_config = super(AddWithFade, self).get_config()
        base_config["fade"] = self.fade
        return base_config

# update the fade parameter in the model
def update_fade(model,newfade):
    for layer in model.layers:
        if isinstance(layer, AddWithFade):
            K.set_value(layer.fade,newfade)

# this has the generator creating realistic deviations across batches
# this is likely more important for faces, etc
class MBStDev(Layer):
    def __init__(self, **kwargs):
        self.smallParam = 1e-8
        super(MBStDev, self).__init__(**kwargs)
    
    # perform the operation
    def call(self, ins):
        # mean value for each pixel across channels
        pixMean = K.mean(ins, axis=0, keepdims=True)
        # standard deviation across each pixel coord (small param regulates singularity)
        stDev = K.sqrt(K.mean(K.square(ins - pixMean), axis=0, keepdims=True)+self.smallParam)
        # mean standard deviation across each pixel coord
        meanStDev = K.mean(stDev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = K.shape(ins)
        outs = K.tile(meanStDev, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        joinedInandOut = K.concatenate([ins, outs], axis=-1)
        return joinedInandOut
    
    # corrects the output shape to match the joint values
    def correct_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)
    
    def get_config(self):
        base_config = super(MBStDev, self).get_config()
        return base_config

# from 1710.10196 - use pixel normalization, a variant of "local response normalization"
# Used to "disallow the scenario where the magnitudes in the generator and discriminator spiral out
# of control as a result of competition" - apply BEFORE activation function in generator only
class PixelNorm(Layer):
    def __init__(self, **kwargs):
        self.smallParam = 1e-8
        super(PixelNorm, self).__init__(**kwargs)
    
    def call(self, ins):
        # -1 is over the filters
        sqPixMean = K.mean(ins**2 + self.smallParam, axis=-1, keepdims=True)
        return ins / K.sqrt(sqPixMean)
    
    def get_config(self):
        base_config = super(PixelNorm, self).get_config()
        return base_config

# this kernel implements equalized learning rate
class Conv2DELR(Conv2D):
    
    def __init__(self, *args, cHe=None, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.), **kwargs):
        if cHe is not None:
            self.c = cHe
        super().__init__(*args, kernel_initializer=kernel_initializer, **kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)
    
    def call(self, inputs):
        if self.rank == 2:
            outputs = K.conv2d(
                               inputs,
                               self.kernel*self.c, # scale kernel
                               strides=self.strides,
                               padding=self.padding,
                               data_format=self.data_format,
                               dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            outputs = K.bias_add(
                                 outputs,
                                 self.bias,
                                 data_format=self.data_format)
        
        if self.activation is not None:
            return self.activation(outputs)
                               return outputs

    def get_config(self):
        base_config = super(Conv2D, self).get_config()
        base_config['cHe'] = self.c
        return base_config
