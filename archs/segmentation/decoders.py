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
                                    Conv2DTranspose,Add
# Attention imports
from tensorflow.keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
# Plot model
from tensorflow.keras.utils import plot_model
from .model_utils import ResidualConcatenation, ResidualLinearBlock, ResidualConvBlock, ResidualBlock


class DecoderBlock(Layer):
    # Parent class for various decoder blocks
    depth_id = 0
    def __init__(self, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        DecoderBlock.depth_id += 1
        self.filters = kwargs.get('filters', None)
        self.drop_rate = kwargs.get('drop_rate', 0.0)
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.strides = kwargs.get('strides', 1)
        self.padding = kwargs.get('padding', 'same')
        self.activation = kwargs.get('activation', 'relu')
        self.depth = kwargs.get('depth', 1)
        self.in_channels = kwargs.get('in_channels', None)
    def call(self, x, **kwargs):
        raise NotImplementedError
    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'padding': self.padding,
                       'activation': self.activation,
                       'depth': self.depth,
                       'drop_rate': self.drop_rate,
                       'in_channels': self.in_channels})
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
        # self.seq = Sequential()
        # if depth>0: # Prepend a residual block
        #     assert in_channels is not None, "in_channels must be specified for depth>0"
        #     self.seq.add(ResidualBlock(filters=in_channels, 
        #                                kernel_size=kernel_size,
        #                                strides=strides,
        #                                padding=padding,
        #                                activation=activation,
        #                                depth=depth,
        #                                drop_rate=drop_rate))
        self.concat = Concatenate()
        self.convT = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', activation=activation)

    def call(self, x,skip, **kwargs):
        x = self.convT(x)
        x = self.concat([x, skip])
        # x = self.seq(x)
        return x
    # def get_config(self):
    #     config = super(DecoderBlockConcat, self).get_config()
    #     config.update({'filters': self.filters,
    #                    'kernel_size': self.kernel_size,
    #                    'strides': self.strides,
    #                    'padding': self.padding,
    #                    'activation': self.activation,
    #                    'depth': self.depth,
    #                    'drop_rate': self.drop_rate,
    #                    'in_channels': self.in_channels})
    #     return config

class DecoderBlockAdd(DecoderBlock):
    """Standard decoder block. Addition followed by transposed convolution"""
    def __init__(self,
                filters:int,
                kernel_size:int=3,
                strides:int=1,
                padding:str="same",
                activation:str="relu",
                depth:int=1,
                drop_rate:float=0.0,
                in_channels:int=None, **kwargs):
        super(DecoderBlockAdd, self).__init__(name=f"decoder_block_add_{DecoderBlock.depth_id}", **kwargs)
        # If the depth is greater than 1 we use a residual addition block to combine the features
        # self.seq = Sequential()
        # if depth>0: # Prepend a residual block
        #     assert in_channels is not None, "in_channels must be specified for depth>0"
        #     self.seq.add(ResidualBlock(filters=in_channels, 
        #                                kernel_size=kernel_size,
        #                                strides=strides,
        #                                padding=padding,
        #                                activation=activation,
        #                                depth=depth,
        #                                drop_rate=drop_rate))
        self.add = Add()
        self.convT = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', activation=activation)

    def call(self, x,skip, **kwargs):
        x = self.convT(x)
        x = self.add([x, skip])
        # x = self.seq(x)
        return x
    # def get_config(self):
    #     config = super(DecoderBlockLinear, self).get_config()
    #     config.update({'filters': self.filters,
    #                    'kernel_size': self.kernel_size,
    #                    'strides': self.strides,
    #                    'padding': self.padding,
    #                    'activation': self.activation,
    #                    'depth': self.depth,
    #                    'drop_rate': self.drop_rate,
    #                    'in_channels': self.in_channels})
    #     return config
class DecoderBlockNoSkip(DecoderBlock):
    """Standard decoder block. Addition followed by transposed convolution"""
    def __init__(self,
                filters:int,
                kernel_size:int=3,
                strides:int=1,
                padding:str="same",
                activation:str="relu",
                depth:int=1,
                drop_rate:float=0.0,
                in_channels:int=None, **kwargs):
        super(DecoderBlockNoSkip, self).__init__(name=f"decoder_block_noskip_{DecoderBlock.depth_id}", **kwargs)
        # If the depth is greater than 1 we use a residual addition block to combine the features
        # self.seq = Sequential()
        # if depth>0: # Prepend a residual block
        #     assert in_channels is not None, "in_channels must be specified for depth>0"
        #     self.seq.add(ResidualBlock(filters=in_channels, 
        #                                kernel_size=kernel_size,
        #                                strides=strides,
        #                                padding=padding,
        #                                activation=activation,
        #                                depth=depth,
        #                                drop_rate=drop_rate))
        self.convT = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', activation=activation)

    def call(self, x, skip=None, **kwargs):
        x = self.convT(x)
        # x = self.seq(x)
        return x


class DecoderBuilder:
    def __init__(self,decoder_type:str, # Type of decoder to use (concat, add)
                num_classes:int, # Number of classes to predict
                filters:List[int],
                kernel_size:List[int]=3,
                strides:List[int]=1,
                padding:List[str]="same",
                activation:List[str]="relu",
                depth:List[int]=1,
                output_depth:int=0,
                output_activation:str="softmax",
                drop_rate:List[float]=0.0,
                encoder_outputs:List[int]=None):
        self.decoder_type = decoder_type
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.depth = depth
        self.output_depth = output_depth
        self.drop_rate = drop_rate
        self.encoder_outputs = encoder_outputs
        # Extract the channels of the encoder outputs
        if encoder_outputs is not None:
            self.in_channels = [x.shape[-1] for x in encoder_outputs[::-1]]
        else:
            self.in_channels = None
        self._make_lists()
        self.output_activation = output_activation
    def _make_lists(self):
        # If the parameters are not lists of same length, extend them with the last value
        if not isinstance(self.filters,list):
            self.filters = [self.filters]
        if not isinstance(self.kernel_size,list):
            self.kernel_size = [self.kernel_size]
        if not isinstance(self.strides,list):
            self.strides = [self.strides]
        if not isinstance(self.padding,list):
            self.padding = [self.padding]
        if not isinstance(self.activation,list):
            self.activation = [self.activation]
        if not isinstance(self.depth,list):
            self.depth = [self.depth]
        if not isinstance(self.drop_rate,list):
            self.drop_rate = [self.drop_rate]
        if not isinstance(self.in_channels,list):
            self.in_channels = [self.in_channels]
        # Extend the lists to the same length, according to the number of filters
        self.kernel_size = self.kernel_size + [self.kernel_size[-1]]*(len(self.filters)-len(self.kernel_size))
        self.strides = self.strides + [self.strides[-1]]*(len(self.filters)-len(self.strides))
        self.padding = self.padding + [self.padding[-1]]*(len(self.filters)-len(self.padding))
        self.activation = self.activation + [self.activation[-1]]*(len(self.filters)-len(self.activation))
        self.depth = self.depth + [self.depth[-1]]*(len(self.filters)-len(self.depth))
        self.drop_rate = self.drop_rate + [self.drop_rate[-1]]*(len(self.filters)-len(self.drop_rate))
        self.in_channels = self.in_channels + [self.in_channels[-1]]*(len(self.filters)-len(self.in_channels))        

    def build(self):
        layers = []
        for i in range(len(self.filters)):
            kwargs = dict(filters=self.filters[i],
                        kernel_size=self.kernel_size[i],
                        strides=self.strides[i],
                        padding=self.padding[i],
                        activation=self.activation[i],
                        depth=self.depth[i],
                        drop_rate=self.drop_rate[i],
                        in_channels=self.in_channels[i])
            if self.decoder_type == "concat":
                layers.append(DecoderBlockConcat(**kwargs))
            elif self.decoder_type == "add":
                layers.append(DecoderBlockAdd(**kwargs))
            elif self.decoder_type == "noskip":
                layers.append(DecoderBlockNoSkip(**kwargs))
            else:
                raise ValueError(f"Decoder type {self.decoder_type} not supported")
            if self.depth[i]>0:
                layers.append(
                    ResidualBlock(filters=self.in_channels[i], 
                                    kernel_size=self.kernel_size[i],
                                    strides=self.strides[i],
                                    padding=self.padding[i],
                                    activation=self.activation[i],
                                    depth=self.depth[i],
                                    drop_rate=self.drop_rate[i])
                )
            else:
                # Identity block
                layers.append(lambda x: x)

        # Finish with a NoSkip block
        self.final = DecoderBlockNoSkip(filters=self.num_classes,
                                        kernel_size=1, 
                                        strides=1, 
                                        padding="same", 
                                        activation=self.output_activation, 
                                        depth=self.output_depth, 
                                        drop_rate=0.0, 
                                        in_channels=self.num_classes)
        self.layers = layers
        self._init_call()
        return self.model
    def _init_call(self):
        encoder_outputs = self.encoder_outputs
        inp = encoder_outputs[-1]
        skips = encoder_outputs[:-1][::-1]
        x = inp
        for i in range(len(self.layers)):
            if i%2==0:
                x = self.layers[i](x,skips[i//2])
            else:
                x = self.layers[i](x)
        self.output = self.final(x)
        self.model = Model(inputs=encoder_outputs[::-1], outputs=self.output, name="Decoder")
    def initialize(self):
        if not hasattr(self,"output"):
            self.model = self.build()
        return self.output

def build_decoder( # Type of decoder to use (concat, add)
                encoder_outputs:List[tf.Tensor],
                num_classes:int,
                filters:List[int],
                kernel_size:List[int]=3,
                strides:List[int]=1,
                padding:List[str]="same",
                activation:List[str]="relu",
                depth:List[int]=1,
                output_depth:int=0,
                output_activation="sigmoid",
                drop_rate:List[float]=0.0,
                decoder_type:str="concat",):

    builder = DecoderBuilder(decoder_type=decoder_type,
                                num_classes=num_classes,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=activation,
                                depth=depth,
                                output_depth=output_depth,
                                output_activation=output_activation,
                                drop_rate=drop_rate,
                                encoder_outputs=encoder_outputs,
                            )
    model = builder.build()
    return model

# def initialize_decoder( # Type of decoder to use (concat, add)
#                 encoder_outputs:List[tf.Tensor],
#                 num_classes:int,
#                 filters:List[int],
#                 kernel_size:List[int]=3,
#                 strides:List[int]=1,
#                 padding:List[str]="same",
#                 activation:List[str]="relu",
#                 depth:List[int]=1,
#                 output_depth=0,
#                 drop_rate:List[float]=0.0,
#                 decoder_type:str="concat",):

#     builder = DecoderBuilder(decoder_type=decoder_type,
#                             num_classes=num_classes,
#                             filters=filters,
#                             kernel_size=kernel_size,
#                             strides=strides,
#                             padding=padding,
#                             activation=activation,
#                             depth=depth,
#                             output_depth=output_depth,
#                             drop_rate=drop_rate,
#                             encoder_outputs=encoder_outputs)
#     return builder.initialize()


        


