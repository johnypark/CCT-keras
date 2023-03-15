# CCT-keras: Compact Transformers implemented in keras
 
Implementation of Compact Transformers from [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

Official Pytorch implementation can be found here: https://github.com/SHI-Labs/Compact-Transformers

## Usage
```python
!pip install git+https://github.com/johnypark/CCT-keras

from CCT_keras import CCT

model = CCT(num_classes = 1000, input_shape = (224, 224, 3))

```
Default CCT() is set as CCT_14_7x2 in the paper, for which the authors used to train on ImageNet from scratch.
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
