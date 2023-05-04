import tensorflow as tf
from tensorflow import keras
from CCT_keras.utils import DropPath
from functools import partial

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
        self.out_dense = keras.layers.Dense(hidden_size, name = "dense_out")
        self.Dropout = keras.layers.Dropout(rate = self.DropOut_rate)
        self.CalcAttention = partial(self.ScaledDotProductAttention, dim  = self.projection_dim)

    def ScaledDotProductAttention(self, query, key, value, dim):
        score = tf.matmul(query, key, transpose_b = True)
        #dim_key = tf.cast(tf.shape(key)[-1], dtype = score.dtype)
        scaled_score = score / tf.math.sqrt(dim)
        weights = tf.nn.softmax(scaled_score, axis = -1)
        weights = self.Dropout(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_to_multihead(self, x, batch_size):
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
        
        query, key, value = [self.separate_to_multihead(tensor, batch_size) for tensor in [query, key, value]]
        
        weighted_value, weights = self.CalcAttention(query, key, value)
        weighted_value = tf.transpose(weighted_value, perm = [0, 2, 1, 3])
        combined_values = tf.reshape(weighted_value, 
                                      shape = (batch_size, -1, self.hidden_size)
                                      )
        output = self.out_dense(combined_values)
        output = self.Dropout(output)
        
        if self.output_weight:
            output = output, weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"DropOut_rate": self.DropOut_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Feed Forward Network (FFN)

class FeedForwardNetwork(keras.layers.Layer):
    def __init__(self, *args, mlp_ratio,
                                DropOut_rate,
                                activation = 'gelu',
                                **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_ratio = mlp_ratio
        self.DropOut_rate = DropOut_rate
        self.activation = activation
    
    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        overhead_dim = int(embedding_dim*self.mlp_ratio)
        self.Dense_hidden = keras.layers.Dense(units = overhead_dim, name = "dense_hidden")
        self.Dense_out = keras.layers.Dense(units = embedding_dim, name = "dense_out")
        self.Activation = keras.layers.Activation(self.activation)
        self.Dropout = keras.layers.Dropout(rate = self.DropOut_rate)
        
    
    def call(self, inputs):
        x = inputs
        x = self.Dense_hidden(x)
        x = self.Activation(x)
        x = self.Dropout(x)
        x = self.Dense_out(x)
        x = self.Activation(x)
        outputs = self.Dropout(x)
        
        return outputs
    
    
    def get_config(self):
        config = super().get_config()
        config.update({"mlp_ratio": self.mlp_ratio})
        config.update({"DropOut_rate": self.DropOut_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Transformer Block

def Transformer_Block(mlp_ratio,
                      num_heads,
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
        mlp = FeedForwardNetwork(mlp_ratio = mlp_ratio,
                      DropOut_rate = DropOut_rate 
		    )(LN_output2)
        if stochastic_depth_rate:
            mlp = DropPath(stochastic_depth_rate)(mlp)
        output = tf.keras.layers.Add()([x1, mlp]) 
                      
        return output
    
    return apply


def MB4D_Block(mlp_ratio,
               embedding_dims,
                      stochastic_depth_rate = None,
                      DropOut_rate = 0.1,
                      activation = 'gelu'):
    def apply(inputs):
        
        x = inputs
        #poolformer layer
        pooling = tf.keras.layers.AveragePooling2D(pool_size =3, 
                                                   strides = 2)(x)
        pooling_output = tf.keras.layers.Add()([inputs, pooling])
        
        #MLP substitude
        x1 = pooling_output
        x1 = keras.layers.Conv2D(
            activation = None,
            filters = embedding_dims*mlp_ratio,
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = keras.layers.BatchNoramlization()(x1)
        x1 = keras.layers.Dropout(DropOut_rate)(x1)
        x1 = keras.layers.Activation(activation)(x1)
        x1 = keras.layers.Conv2D(
            activation = None,
            filters = embedding_dims,
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = keras.layers.BatchNoramlization()(x1)
        x1 = keras.layers.Dropout(DropOut_rate)(x1)
        if stochastic_depth_rate:
            x1 = DropPath(stochastic_depth_rate)(x1)
        
        output = tf.keras.layers.Add()([pooling_output, x1])     
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

class add_positional_embedding(keras.layers.Layer):
    
    def __init__(self, 
                 #num_patches, 
                 #embedding_dim,
                 embedding_type = 'learnable',
                 noise_stddev = 2e-1):
        super().__init__()
        
        self.embedding_type = embedding_type
        #self.num_patches = num_patches # this can be removed
        #self.embedding_dim = embedding_dim # this can be removed
        self.noise_stddev = noise_stddev
        
    def build(self, input_shape):
        assert (
            len(input_shape)==3 
        ), "Expected tensor dim=3. Got {}".format(len(input_shape))
            
        num_patches = input_shape[-2]
        embedding_dim = input_shape[-1]
        if self.embedding_type:
            if self.embedding_type == 'sinusodial':
                self.positional_embedding = tf.Variable(sinusodial_embedding(num_patches = num_patches,
                                                embedding_dim = embedding_dim),
                                                name ='sinosodial'
                                                ),
                        trainable = False)
            elif self.embedding_type == 'learnable':
                    self.positional_embedding = tf.Variable(
                        tf.random.truncated_normal(shape=[1, num_patches, embedding_dim], stddev= self.noise_stddev),
                        trainable = True,
                        name = 'learnable')
                
        else: # else simple gaussian noise injection
                
            noise = tf.random_normal_initializer(stddev = self.noise_stddev) 
            self.positional_embedding = tf.Variable(
                    noise(shape = [1, num_patches, embedding_dim]),
                    trainable = False,
                    name = 'gaussian_noise')
            
    def call(self, input):
        PE = tf.cast(self.positional_embedding, dtype = input.dtype)
        input = tf.math.add(input, PE)
        return input