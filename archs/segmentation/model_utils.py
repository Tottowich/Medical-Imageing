# TensorFlow basic imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
# Model specific imports
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Activation, Dropout, Dense,Flatten, SpatialDropout2D, UpSampling2D
# Attention imports
from tensorflow.keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
# Plot model
from tensorflow.keras.utils import plot_model

class ResidualAddition(Layer):
    """Residual addition wrapper layer"""
    def __init__(self, fn,activation,skip_connection=None,**kwargs):
        super(ResidualAddition, self).__init__(**kwargs)
        self.add = keras.layers.Add()
        self.activation = Activation(activation)
        self.fn = fn
        self.skip_connection = skip_connection if skip_connection else lambda x: x
    def call(self, x, **kwargs):
        return self.activation(self.add([self.skip_connection(x), self.fn(x)]))
    def get_config(self):
        config = super(ResidualAddition, self).get_config()
        config.update({'fn': self.fn})
        return config


class ResidualConcatenation(Layer):
    """Residual concatenation wrapper layer"""
    def __init__(self, fn,**kwargs):
        super(ResidualConcatenation, self).__init__(**kwargs)
        self.concat = keras.layers.Concatenate()
        self.fn = fn
    def call(self, x, **kwargs):
        return self.concat([x, self.fn(x)])
    def get_config(self):
        config = super(ResidualConcatenation, self).get_config()
        config.update({'fn': self.fn})
        return config

class ResidualLinearBlock(Layer):
    """Residual linear block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ResidualLinearBlock, self).__init__(**kwargs)
        self.seq = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            Activation(activation),
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            Activation(activation)
        ])
        self.res_add = ResidualAddition(self.seq,activation=activation)

    def call(self, x, **kwargs):
        return self.res_add(x)
    def get_config(self):
        config = super(ResidualLinearBlock, self).get_config()
        config.update({'seq': self.seq})
        return config

class ResidualConvBlock(Layer):
    """Residual convolution block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.seq = Sequential([
            # Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False),
            # BatchNormalization(),
            # Activation(activation),
            ConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation),
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False),
            BatchNormalization(),
            # Activation(activation)
        ])
        skip = Sequential([
            Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding, use_bias=False),
            BatchNormalization()
        ])

        self.res_add = ResidualAddition(self.seq,activation=activation,skip_connection=skip)
    def call(self, x, **kwargs):
        return self.res_add(x) #WRONG
    def get_config(self):
        config = super(ResidualConvBlock, self).get_config()
        config.update({'seq': self.seq})
        config.update({'conv': self.conv})
        return config

class ConvBlock(Layer):
    """Convolution block"""
    def __init__(self, filters:int, kernel_size:int=3, strides:int=1, padding:str="same", activation:str="relu", **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.seq = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=False),
            BatchNormalization(),
            Activation(activation)
        ])
    def call(self, x, **kwargs):
        return self.seq(x)
    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({'seq': self.seq})
        return config

class ResidualBlock(Layer):
    """Encoder block"""
    def __init__(self,
                filters:int,
                kernel_size:int=3,
                strides:int=1,
                padding:str="same",
                activation:str="relu",
                depth:int=1,
                drop_rate:float=0.0, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.seq = Sequential()
        for i in range(depth):
            if i==0:
                self.seq.add(ResidualConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
            else:
                self.seq.add(ResidualLinearBlock(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
            if drop_rate>0.0:
                self.seq.add(SpatialDropout2D(drop_rate))
    def call(self, x, **kwargs):
        return self.seq(x)
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({'seq': self.seq})
        return config