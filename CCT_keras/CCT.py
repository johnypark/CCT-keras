# CCT: Escaping the Big Data Paradigm with Compact Transformers
# Paper: https://arxiv.org/pdf/2104.05704.pdf
# CCT-L/KxT: 
# K transformer encoder layers 
# T-layer convolutional tokenizer with KxK kernel size.
# In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
# CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from CCT_keras.building_blocks import add_positional_embedding, Transformer_Block
from CCT_keras.Tokenizer import *

settings = dict()

settings['epsilon'] = 1e-6
settings['denseInitializer'] = 'glorot_uniform'
settings['conv2DInitializer'] = 'he_normal'

# Sequence Pooling with additional channel option
def SeqPool(settings, n_attn_channel = 1): 
    """ Learnable pooling layer. Replaces the class token in ViT.
    In the paper they tested static pooling methods but learnable weighting is more effcient, 
    because each embedded patch does not contain the same amount of entropy. 
    Enables the model to apply weights to tokens with repsect to the relevance of their information
    """
    def apply(inputs):
        x = inputs    
        x = tf.keras.layers.LayerNormalization(
            epsilon = settings['epsilon'],
        )(x)
        x_init = x
        x = tf.keras.layers.Dense(units = n_attn_channel, activation = 'softmax')(x)
        w_x = tf.matmul(x, x_init, transpose_a = True)
        w_x = tf.keras.layers.Flatten()(w_x)     
        return w_x

    return apply
        
        
def get_dim_Conv_Tokenizer(Conv_strides, 
                           pool_strides, 
                           num_tokenizer_ConvLayers):

    def apply(dim):
        start = dim
        for k in range(num_tokenizer_ConvLayers):
            Conv_out_dim = -(start // -Conv_strides)            
            pool_out_dim = -(Conv_out_dim // - pool_strides)  
            start = pool_out_dim          
        return pool_out_dim
    return apply
        
### CCT MODEL
def CCT(num_classes, 
        input_shape = (None, None, 3),
        num_TransformerLayers = 14,
        num_heads = 6,
        mlp_ratio = 3,
        embedding_dim = 384,
        tokenizer_kernel_size = 7,
        tokenizer_strides = 2,
        num_tokenizer_ConvLayers = 2,
        DropOut_rate = 0.1,
        stochastic_depth_rate = 0.1,
        settings = settings,
        n_SeqPool_weights = 1,
        positional_embedding = 'learnable',
        add_top = True,
        final_DropOut_rate = 0.3):

    """ CCT-L/KxC: L transformer encoder layers and K Conv2D Kernel, with C Conv2D layers.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    Default settings are set for CCT-14/7x2
    embedding_type: learnable or sinusodial
    """
    Tokenizer_ConvLayers_dims = [embedding_dim//2**(i) for i in reversed(range(num_tokenizer_ConvLayers))]
    # Need to add tokenizer settings
    input = tf.keras.layers.Input(
		shape = input_shape)
    
    x = input
    
    ### Tokenize Image
    x = Conv_Tokenizer(strides = tokenizer_strides, 
              kernel_size = tokenizer_kernel_size,
              kernel_initializer = settings['conv2DInitializer'],
              activation = 'relu',
              pool_size = 3,
              pooling_stride = 2,
              list_embedding_dims = Tokenizer_ConvLayers_dims)(x)
    
    ### Add Positional Embedding
    if positional_embedding:
        x = add_positional_embedding(embedding_type = positional_embedding)(x)    
    x = tf.keras.layers.Dropout(rate = DropOut_rate)(x)
    
    ### Transformer Blocks
    TFL = dict()
    TFL[0] = x
    for L in range(num_TransformerLayers):
        TFL[L+1] = Transformer_Block(mlp_ratio = mlp_ratio,
                      num_heads = num_heads,
                      DropOut_rate = DropOut_rate,
                      stochastic_depth_rate = stochastic_depth_rate,
                      LayerNormEpsilon = settings['epsilon'],
                      )(TFL[L])
        
    ### Sequence Pooling ####
    penultimate = SeqPool(settings = settings,
                     n_attn_channel = n_SeqPool_weights)(TFL[num_TransformerLayers])
    
    if add_top:
        penultimate = tf.keras.layers.Dropout(final_DropOut_rate)(penultimate)
    
        ### Classification Head
        outputs = tf.keras.layers.Dense(
            activation = 'softmax',
            kernel_initializer = settings['denseInitializer'],
            units = num_classes,
            use_bias = True
        )(penultimate)
        
    else:
        outputs = penultimate
    
    return tf.keras.Model(inputs = input, outputs = outputs)

    
### CCT MODEL
def CCTV2_7_3x1(num_classes, 
        input_shape = (None, None, 3),
        num_TransformerLayers = 7,
        num_heads = 4,
        mlp_ratio = 2,
        embedding_dim = 196,
        tokenizer_kernel_size = 3,
        tokenizer_strides = 2,
        num_tokenizer_ConvLayers = 1,
        DropOut_rate = 0.1,
        stochastic_depth_rate = 0.1,
        settings = settings,
        n_SeqPool_weights = 2,
        positional_embedding = 'learnable',
        add_top = True,
        final_DropOut_rate = 0.3):

    """ CCT-L/KxC: L transformer encoder layers and K Conv2D Kernel, with C Conv2D layers.
    In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
    CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification
    Default settings are set for CCT-14/7x2
    embedding_type: learnable or sinusodial
    """
    Tokenizer_ConvLayers_dims = [embedding_dim//2**(i) for i in reversed(range(num_tokenizer_ConvLayers))]
    # Need to add tokenizer settings
    input = tf.keras.layers.Input(
		shape = input_shape)
    
    x = input
    
    ### Tokenize Image
    x = Conv_TokenizerV2(strides = tokenizer_strides, 
              kernel_size = tokenizer_kernel_size,
              mlp_ratio = mlp_ratio,
              activation = 'relu',
              pool_size = 3,
              pooling_stride = 2,
              list_embedding_dims = Tokenizer_ConvLayers_dims)(x)
    
    ### Add Positional Embedding
    if positional_embedding:
        x = add_positional_embedding(embedding_type = positional_embedding)(x)    
    x = tf.keras.layers.Dropout(rate = DropOut_rate)(x)
    
    ### Transformer Blocks
    TFL = dict()
    TFL[0] = x
    for L in range(num_TransformerLayers):
        TFL[L+1] = Transformer_Block(mlp_ratio = mlp_ratio,
                      num_heads = num_heads,
                      DropOut_rate = DropOut_rate,
                      stochastic_depth_rate = stochastic_depth_rate,
                      LayerNormEpsilon = settings['epsilon'],
                      )(TFL[L])
        
    ### Sequence Pooling ####
    penultimate = SeqPool(settings = settings,
                     n_attn_channel = n_SeqPool_weights)(TFL[num_TransformerLayers])
    
    if add_top:
        penultimate = tf.keras.layers.Dropout(final_DropOut_rate)(penultimate)
    
        ### Classification Head
        outputs = tf.keras.layers.Dense(
            activation = 'softmax',
            kernel_initializer = settings['denseInitializer'],
            units = num_classes,
            use_bias = True
        )(penultimate)
        
    else:
        outputs = penultimate
    
    return tf.keras.Model(inputs = input, outputs = outputs)


def CCT_4(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_7(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_7_3x1(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_7_3x1_32_C100(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_7_3x1_32_sine_c100(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_7_7x2_224_sine(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_14(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_14_7x2(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_14_7x2_224(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)

def CCT_14_7x2_384(inputs):
    outputs = inputs
    return tf.keras.Model(inputs = input, outputs = outputs)
