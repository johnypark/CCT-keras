## CCT-keras: Compact Transformers implemented in keras
 
Implementation of Compact Transformers from [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

The official Pytorch implementation can be found here: https://github.com/SHI-Labs/Compact-Transformers

## Overview 
Compact Convolutional Transformer (CCT) is represented by three main changes on ViT:
- Convolutional Tokenizer, instead of the direct image patching of ViT
- Sequence Pooling instead of the Class Token
- Learnable Positional Embedding instead of Sinusodial Embedding

CCT naturally inherits other components of ViT, such as:
- Multi-Head Self Attention
- Feed Forward Network (MLP Block)
- Dropouts and Stochastic Depth

## Usage
```python
!pip install git+https://github.com/johnypark/CCT-keras

from CCT_keras import CCT

model = CCT(num_classes = 1000, input_shape = (224, 224, 3))

```
The default CCT() is set as CCT_14_7x2 in the paper, for which the authors used to train on ImageNet from scratch.
```

model = summary()
.
.
.
layer_normalization_55 (LayerN  (None, 196, 384)    768         ['add_53[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 multi_head_self_attention_27 (  (None, None, 384)   591360      ['layer_normalization_55[0][0]'] 
 MultiHeadSelfAttention)                                                                          
                                                                                                  
 drop_path_54 (DropPath)        (None, None, 384)    0           ['multi_head_self_attention_27[0]
                                                                 [0]']                            
                                                                                                  
 add_54 (Add)                   (None, 196, 384)     0           ['add_53[0][0]',                 
                                                                  'drop_path_54[0][0]']           
                                                                                                  
 layer_normalization_56 (LayerN  (None, 196, 384)    768         ['add_54[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 dense_56 (Dense)               (None, 196, 1152)    443520      ['layer_normalization_56[0][0]'] 
                                                                                                  
 activation_54 (Activation)     (None, 196, 1152)    0           ['dense_56[0][0]']               
                                                                                                  
 dropout_57 (Dropout)           (None, 196, 1152)    0           ['activation_54[0][0]']          
                                                                                                  
 dense_57 (Dense)               (None, 196, 384)     442752      ['dropout_57[0][0]']             
                                                                                                  
 activation_55 (Activation)     (None, 196, 384)     0           ['dense_57[0][0]']               
                                                                                                  
 dropout_58 (Dropout)           (None, 196, 384)     0           ['activation_55[0][0]']          
                                                                                                  
 drop_path_55 (DropPath)        (None, 196, 384)     0           ['dropout_58[0][0]']             
                                                                                                  
 add_55 (Add)                   (None, 196, 384)     0           ['add_54[0][0]',                 
                                                                  'drop_path_55[0][0]']           
                                                                                                  
 layer_normalization_57 (LayerN  (None, 196, 384)    768         ['add_55[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 dense_58 (Dense)               (None, 196, 1)       385         ['layer_normalization_57[0][0]'] 
                                                                                                  
 tf.linalg.matmul_1 (TFOpLambda  (None, 1, 384)      0           ['dense_58[0][0]',               
 )                                                                'layer_normalization_57[0][0]'] 
                                                                                                  
 flatten_1 (Flatten)            (None, 384)          0           ['tf.linalg.matmul_1[0][0]']     
                                                                                                  
 dropout_59 (Dropout)           (None, 384)          0           ['flatten_1[0][0]']              
                                                                                                  
 dense_59 (Dense)               (None, 1000)         385000      ['dropout_59[0][0]']             
                                                                                                  
==================================================================================================
Total params: 24,735,401
Trainable params: 24,735,401
Non-trainable params: 0
__________________________________________________________________________________________________

```

## Access Model Weights

```python

model_weights_dict = {(w.name): (idx, w.dtype, w.shape) for idx, w in enumerate(model.weights)}
names_dense = [name for name in model_weights_dict.keys() if 'dense' in name]
idx_dense = [model_weights_dict[name][0] for name in names_dense]


>>model_weights_dict
{'conv2d/kernel:0': (0, tf.float32, TensorShape([3, 3, 3, 98])),
 'conv2d_1/kernel:0': (1, tf.float32, TensorShape([3, 3, 98, 196])),
 'layer_normalization/gamma:0': (2, tf.float32, TensorShape([196])),
 'layer_normalization/beta:0': (3, tf.float32, TensorShape([196])),
 'multi_head_self_attention/query/kernel:0': (4,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention/query/bias:0': (5, tf.float32, TensorShape([196])),
 'multi_head_self_attention/key/kernel:0': (6,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention/key/bias:0': (7, tf.float32, TensorShape([196])),
 'multi_head_self_attention/value/kernel:0': (8,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention/value/bias:0': (9, tf.float32, TensorShape([196])),
 'multi_head_self_attention/out/kernel:0': (10,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention/out/bias:0': (11, tf.float32, TensorShape([196])),
 'layer_normalization_1/gamma:0': (12, tf.float32, TensorShape([196])),
 'layer_normalization_1/beta:0': (13, tf.float32, TensorShape([196])),
 'dense/kernel:0': (14, tf.float32, TensorShape([196, 392])),
 'dense/bias:0': (15, tf.float32, TensorShape([392])),
 'dense_1/kernel:0': (16, tf.float32, TensorShape([392, 196])),
 'dense_1/bias:0': (17, tf.float32, TensorShape([196])),
 'layer_normalization_2/gamma:0': (18, tf.float32, TensorShape([196])),
 'layer_normalization_2/beta:0': (19, tf.float32, TensorShape([196])),
 'multi_head_self_attention_1/query/kernel:0': (20,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention_1/query/bias:0': (21,
  tf.float32,
  TensorShape([196])),
 'multi_head_self_attention_1/key/kernel:0': (22,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention_1/key/bias:0': (23,
  tf.float32,
  TensorShape([196])),
 'multi_head_self_attention_1/value/kernel:0': (24,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention_1/value/bias:0': (25,
  tf.float32,
  TensorShape([196])),
 'multi_head_self_attention_1/out/kernel:0': (26,
  tf.float32,
  TensorShape([196, 196])),
 'multi_head_self_attention_1/out/bias:0': (27,
  tf.float32,
  TensorShape([196])),
 'layer_normalization_3/gamma:0': (28, tf.float32, TensorShape([196])),
 'layer_normalization_3/beta:0': (29, tf.float32, TensorShape([196])),
 'dense_2/kernel:0': (30, tf.float32, TensorShape([196, 392])),
 'dense_2/bias:0': (31, tf.float32, TensorShape([392])),
 'dense_3/kernel:0': (32, tf.float32, TensorShape([392, 196])),
 'dense_3/bias:0': (33, tf.float32, TensorShape([196])),
 'layer_normalization_4/gamma:0': (34, tf.float32, TensorShape([196])),
 'layer_normalization_4/beta:0': (35, tf.float32, TensorShape([196])),

```
