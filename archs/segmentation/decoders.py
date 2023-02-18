# This file contains the decoder classes for the segmentation models.

# The decoders are used to upsample the low level features from the encoder, creating a high level semantic representation of the image.

# The decoders are used in the U-Net architecture.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
# Model specific imports
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, \
                                    Activation, Dropout, Dense,Flatten, SpatialDropout2D, UpSampling2D, \
                                    Conv2DTranspose
# Attention imports
from tensorflow.keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
# Plot model
from tensorflow.keras.utils import plot_model
from .model_utils import ResidualAddition, ResidualConcatenation, ResidualLinearBlock, ResidualConvBlock, ResidualBlock


class DecoderBlock(Layer):
    # Parent class for various decoder blocks
    depth_id = 0
    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        DecoderBlock.depth_id += 1
    def call(self, x, **kwargs):
        raise NotImplementedError
    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        return config

class DecoderBlockConcat(DecoderBlock):
    """Standard decoder block. Concatenation followed by transposed convolution"""
    def __init__(self,
                filters:int,
                kernel_size:int=3,
                strides:int=1,
                padding:str="same",
                activation:str="relu",
                depth:int=1,
                drop_rate:float=0.0,
                in_channels:int=None, **kwargs):
        super(DecoderBlockConcat, self).__init__(name=f"decoder_block_concat_{DecoderBlock.depth_id}", **kwargs)
        # If the depth is greater than 1 we use a residual addition block to combine the features
        self.seq = Sequential()
        if depth>0: # Prepend a residual block
            assert in_channels is not None, "in_channels must be specified for depth>0"
            self.seq.add(ResidualBlock(filters=in_channels, 
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       activation=activation,
                                       depth=depth,
                                       drop_rate=drop_rate))
        self.concat = Concatenate()
        self.conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='valid', activation=activation)
    def call(self, x,skip, **kwargs):
        x = self.seq(x)
        x = self.concat([x, skip])
        x = self.conv(x)
        return x
    def get_config(self):
        config = super(DecoderBlockConcat, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'padding': self.padding,
                       'activation': self.activation,
                       'depth': self.depth,
                       'drop_rate': self.drop_rate,
                       'in_channels': self.in_channels})
        return config

        


