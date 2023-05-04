#!python

# Patch Generators        
import tensorflow as tf
import tensorflow.keras as keras
from CCT_keras.utils import DropPath

def Conv_Tokenizer(
              kernel_size,
              strides = 2, 
              ##kernel_initializer,
              activation = 'relu',
              list_embedding_dims = [256], 
              pool_size = 3,
              pooling_stride = 2,
              name = None,
              padding = 'same',
              use_bias = False,
              **kwargs):
  
    def apply(inputs):
        
        x = inputs
        num_conv_tokenizers = len(list_embedding_dims)
        for k in range(num_conv_tokenizers):
            x = keras.layers.Conv2D(
            activation = activation,
            filters = list_embedding_dims[k],
            kernel_size = kernel_size,
            strides = strides,
            #kernel_initializer = kernel_initializer,
            name = name,
            padding = padding,
            use_bias = use_bias,
            **kwargs
            )(x)
            x = keras.layers.MaxPool2D(
            #name = name+"maxpool_1",
            pool_size = pool_size, 
            strides = pooling_stride,
            padding = padding
            )(x)
        x =  tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])
        
        return x

    return apply


def Conv_TokenizerV2(
              kernel_size,
              mlp_ratio,
              strides = 2,
              activation = 'relu',
              list_embedding_dims = [256], 
              pool_size = 3,
              pooling_stride = 2,
              name = None,
              padding = 'same',
              use_bias = False,
              DropOut_rate = 0.1,
              stochastic_depth_rate = 0.1,
              **kwargs):
  
    def apply(inputs):
        
        x = inputs
        num_conv_tokenizers = len(list_embedding_dims)
        for k in range(num_conv_tokenizers):
            x = keras.layers.Conv2D(
            activation = activation,
            filters = list_embedding_dims[k],
            kernel_size = kernel_size,
            strides = strides,
            #kernel_initializer = kernel_initializer,
            name = name,
            padding = padding,
            use_bias = use_bias,
            **kwargs
            )(x)
            
            
        #poolformer layer
        pooling = tf.keras.layers.AveragePooling2D(pool_size = pool_size, 
                                                   strides = pooling_stride)(x)
        pooling_output = tf.keras.layers.Add()([x, pooling])
        
        #MLP substitude
        x1 = pooling_output
        x1 = keras.layers.Conv2D(
            activation = None,
            filters = list_embedding_dims[-1]*mlp_ratio,
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = keras.layers.BatchNoramlization()(x1)
        x1 = keras.layers.Dropout(DropOut_rate)(x1)
        x1 = keras.layers.Activation(activation)(x1)
        x1 = keras.layers.Conv2D(
            activation = None,
            filters = list_embedding_dims[-1],
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = keras.layers.BatchNoramlization()(x1)
        x1 = keras.layers.Dropout(DropOut_rate)(x1)
        if stochastic_depth_rate:
            x1 = DropPath(stochastic_depth_rate)(x1)
        
        x2 = tf.keras.layers.Add()([pooling_output, x1])     
            
        output =  tf.reshape(x, shape=[tf.shape(x2)[0], -1, tf.shape(x2)[-1]])
        
        return output

    return apply

