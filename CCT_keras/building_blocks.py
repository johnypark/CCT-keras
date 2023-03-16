__copyright__ = """
Building Blocks CCT-keras
Copyright (c) 2023 John Park
"""
### style adapted from TensorFlow authors

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from einops import rearrange
from CCT_keras.utils import DropPath

# MHSA layer 
# Adopted from: https://github.com/faustomorales/vit-keras/blob/master/vit_keras/utils.py
# Also learn: https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, *args, num_heads, DropOut_rate = 0.1, output_weight = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.output_weight = output_weight
        self.DropOut_rate = DropOut_rate

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads !=0:
          raise ValueError(
              f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
              )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = keras.layers.Dense(hidden_size, name = "dense_query")
        self.key_dense = keras.layers.Dense(hidden_size, name = "dense_key")
        self.value_dense = keras.layers.Dense(hidden_size, name = "dense_value")
        self.combine_heads = keras.layers.Dense(hidden_size, name = "dense_out")
        self.Dropout = keras.layers.Dropout(rate = self.DropOut_rate)

    def ScaledDotProductAttention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b = True)
        dim_key = tf.cast(tf.shape(key)[-1], dtype = score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis = -1)
        weights = self.Dropout(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
                      tensor = x, 
                      shape = (batch_size, -1, self.num_heads, self.projection_dim)
                      )
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        weighted_value, weights = self.ScaledDotProductAttention(query, key, value)
        weighted_value = tf.transpose(weighted_value, perm = [0, 2, 1, 3])
        multihead_values = tf.reshape(weighted_value, 
                                      shape = (batch_size, -1, self.hidden_size)
                                      )
        output = self.combine_heads(multihead_values)
        output = self.Dropout(output)
        
        if self.output_weight:
            output = output, weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Feed Forward Network (FFN)

def MLP_block(embedding_dim,
              mlp_ratio,
              DropOut_rate,
              activation = 'gelu',
              name = None):
    
    def apply(inputs):
        x = inputs
        x = keras.layers.Dense(units = int(embedding_dim*mlp_ratio))(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(rate = DropOut_rate)(x)
        x = keras.layers.Dense(units = embedding_dim)(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(rate = DropOut_rate)(x)
        
        return x 
    
    return apply

# Transformer Block

def Transformer_Block(mlp_ratio,
                      num_heads,
                      projection_dims,
                      stochastic_depth_rate = None,
                      DropOut_rate = 0.1,
                      LayerNormEpsilon = 1e-6):
    def apply(inputs):
        
        x = inputs
        #Attention
        LN_output1 = tf.keras.layers.LayerNormalization(
			epsilon = LayerNormEpsilon
		    )(inputs)
        att = MultiHeadSelfAttention(
			num_heads = num_heads,
            DropOut_rate = DropOut_rate
			)(LN_output1)
        if stochastic_depth_rate:
            att = DropPath(stochastic_depth_rate)(att)
        att_output = tf.keras.layers.Add()([x, att])
        
        #Feed Forward Network
        x1 = att_output
        LN_output2 = tf.keras.layers.LayerNormalization(
            epsilon = LayerNormEpsilon
            )(att_output)
        mlp = MLP_block(embedding_dim = projection_dims,
                            mlp_ratio = mlp_ratio,
                      DropOut_rate = DropOut_rate 
		    )(LN_output2)
        if stochastic_depth_rate:
            mlp = DropPath(stochastic_depth_rate)(mlp)
        output = tf.keras.layers.Add()([x1, mlp]) 
                      
        return output
    
    return apply
    
# Positional embedding

def sinusodial_embedding(num_patches, embedding_dim):
    
        """ sinusodial embedding in the attention is all you need paper 
        example:
        >> plt.imshow(sinusodial_embedding(100,120).numpy()[0], cmap='hot',aspect='auto')
        """
    
        def criss_cross(k):
            n_even = k - k//2
            even = list(range(n_even))
            odd = list(range(n_even, k))
            ccl = []
            for i in range(k//2):
                ccl = ccl+ [even[i]]+ [odd[i]]
            if k//2 != k/2:
                ccl = ccl + [even[k//2]]
            return ccl
            
        embed = tf.cast(([[p / (10000 ** (2 * (i//2) / embedding_dim)) for i in range(embedding_dim)] for p in range(num_patches)]), tf.float32)
        even_col =  tf.sin(embed[:, 0::2])
        odd_col = tf.cos(embed[:, 1::2])
        embed = tf.concat([even_col, odd_col], axis = 1)
        embed = tf.gather(embed, criss_cross(embedding_dim), axis = 1)
        embed = tf.expand_dims(embed, axis=0)

        return embed

class add_positional_embedding():
    
    def __init__(self, 
                 num_patches, 
                 embedding_dim,
                 embedding_type = 'sinusodial'):
        
        self.embedding_type = embedding_type
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        if embedding_type:
            if embedding_type == 'sinusodial':
                self.positional_embedding = tf.Variable(sinusodial_embedding(num_patches = self.num_patches,
                                              embedding_dim = self.embedding_dim
                                              ),
                    trainable = False)
            elif embedding_type == 'learnable':
                self.positional_embedding = tf.Variable(tf.random.truncated_normal(shape=[1, self.num_patches, self.embedding_dim], stddev=0.2))
            
        else:
            self.positional_embedding = None
        
    def __call__(self, input):
        input = tf.math.add(input, self.positional_embedding)
        return input