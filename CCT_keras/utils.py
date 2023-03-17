#!python

import tensorflow as tf
from tensorflow import keras 

class DropPath(keras.layers.Layer):
    """ DropPath class which adopted ideas from the Pytorch DropPath 
    https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py
    Tested generating mask with from a bernoulli distribution, which turns out to 
    disrupt training for some unknown reason.
    This version of the DropPath layer wraps around utilizes keras.layers.Dropout directly and works well.
    Args:
        rate (_type_): the rate at which DropPath  
    """
    def __init__(self, rate, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        num_shapes = len(input_shape)
        shape = (None,)+(1,)*(num_shapes-1)
        self.StochasticDrop = keras.layers.Dropout(self.rate, noise_shape = shape)
        self.Identity = keras.layers.Identity(trainable=False, dynamic=False)
        
    def call(self, inputs, training = None):
        if self.rate == 0:
            return self.Identity(inputs)    
        return self.StochasticDrop(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_rate": self.rate}
        return {**base_config, **config}
    