# This file contains the encoder classes for the segmentation models of brain tumor segmentation.
# The encoders are used to extract features from the input image, creating a low level representation of the image.
# The encoders are used in the U-Net architecture.
# Some encoders contain attention layers, which are used to focus on the most important parts of the image.

# TensorFlow basic imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
# Model specific imports
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Activation, Dropout, Dense,Flatten
# Attention imports
from tensorflow.keras.layers import Attention, MultiHeadAttention
# Other imports
import numpy as np
from typing import List, Tuple, Dict, Union, Optional


class PatchEmbedding(Layer):
    # Patch/Position embedding from the paper: An image is worth 16x16 words: Transformers for image recognition at scale
    def __init__(self, patch_size:int=16, embed_dim:int=768, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        # Input image has size: (batch_size, height, width, channels)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Patch the image into patches using a convolution
        self.patch_conv = Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size)
        # Rearrange the patches into: (batch_size, num_patches, embed_dim)
        self.reshape = keras.layers.Reshape((-1, embed_dim))
        # Add a position embedding and a class token

    def call(self, x:tf.Tensor, **kwargs)->tf.Tensor:
        x = self.patch_conv(x)
        x = self.reshape(x)
        return x





class VisualAttention(Model):
    """Visual attention from the paper: An image is worth 16x16 words: Transformers for image recognition at scale"""
    def __init__(self, d_model:int, num_heads:int=8, **kwargs):
        super(VisualAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = keras.layers.Dense(d_model)

def test_build_model():
    # Test patch embedding
    patch_embedding = PatchEmbedding(patch_size=16, embed_dim=768)
    print(patch_embedding)
    inp = Input(shape=(256, 256, 3))
    x = patch_embedding(inp)
    print(x.shape)
if __name__=="__main__":
    test_build_model()