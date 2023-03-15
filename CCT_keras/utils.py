#!python

import tensorflow as tf
from tensorflow import keras 

class DropPath(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        num_shapes = len(input_shape)
        shape = (None,)+(1,)*(num_shapes-1)
        self.StochasticDrop = keras.layers.Dropout(self.rate, noise_shape = shape)
        
    def call(self, inputs, training = None):
        return self.StochasticDrop(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_rate": self.rate}
        return {**base_config, **config}
    