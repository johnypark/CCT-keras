#!python

# Patch Generators        
import tensorflow as tf
import tensorflow.keras as keras
import einops


class extract_by_size():
    def __init__(self, patch_size, padding = 'VALID'):
        self.patch_size = patch_size
        self.padding = padding
        
    def __call__(self, input):
        x = tf.image.extract_patches( images = input, 
                                  sizes = [1, self.patch_size, self.patch_size, 1],
                                  strides = [1, self.patch_size, self.patch_size, 1],
                                  rates = [1, 1, 1, 1],
                                  padding = self.padding
                                  )
        return x


class extract_by_patch():
    def __init__(self, n_patches, padding = 'VALID'):
        self.n_patches = n_patches
        self.padding = padding

    def get_overlap(self, image_size, n_patches):
        from math import ceil
        n_overlap = n_patches - 1
        patch_size = ceil(image_size/ n_patches)
        return patch_size, (n_patches*patch_size - image_size) // n_overlap
    
  
    def __call__(self, inputs):
        patch_size_x, overlap_x = self.get_overlap(image_size = tf.shape(inputs)[1], n_patches = self.n_patches )
        patch_size_y, overlap_y = self.get_overlap(image_size = tf.shape(inputs)[2], n_patches = self.n_patches )
    
        result = tf.image.extract_patches(images = inputs,
                           sizes=[1, patch_size_x, patch_size_y, 1],
                           strides=[1, (patch_size_x - overlap_x), (patch_size_y - overlap_y), 1],
                           rates=[1, 1, 1, 1],
                           padding= self.padding)
        return result


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
        #strides = strides if strides is not None else max(1, (kernel_size // 2) - 1)
        #padding = padding if padding is not None else max(1, (kernel_size // 2))
    
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
