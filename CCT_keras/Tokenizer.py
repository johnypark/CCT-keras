#!python

# Patch Generators        
import tensorflow as tf
import tensorflow.keras as keras

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
