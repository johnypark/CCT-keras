# CCT-keras: Compact Transformers implemented in keras
 
Implementation of Compact Transformers from [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

Official Pytorch implementation can be found here: https://github.com/SHI-Labs/Compact-Transformers

## Usage
```python
!pip install git+https://github.com/johnypark/CCT-keras

from CCT_keras import CCT

model = CCT(num_classes = 100, input_shape = (224, 224, 3))

```
Default CCT() is set as CCT_14_7x2, which is for training on ImageNet:
```

model = summary()
.
.
.

 activation_27 (Activation)     (None, 196, 384)     0           ['dense_27[0][0]']               
                                                                                                  
 dropout_28 (Dropout)           (None, 196, 384)     0           ['activation_27[0][0]']          
                                                                                                  
 drop_path_27 (DropPath)        (None, 196, 384)     0           ['dropout_28[0][0]']             
                                                                                                  
 add_27 (Add)                   (None, 196, 384)     0           ['add_26[0][0]',                 
                                                                  'drop_path_27[0][0]']           
                                                                                                  
 layer_normalization_28 (LayerN  (None, 196, 384)    768         ['add_27[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 dense_28 (Dense)               (None, 196, 1)       385         ['layer_normalization_28[0][0]'] 
                                                                                                  
 tf.linalg.matmul (TFOpLambda)  (None, 1, 384)       0           ['dense_28[0][0]',               
                                                                  'layer_normalization_28[0][0]'] 
                                                                                                  
 flatten (Flatten)              (None, 384)          0           ['tf.linalg.matmul[0][0]']       
                                                                                                  
 dropout_29 (Dropout)           (None, 384)          0           ['flatten[0][0]']                
                                                                                                  
 dense_29 (Dense)               (None, 100)          38500       ['dropout_29[0][0]']             
                                                                                                  
==================================================================================================
Total params: 24,388,901
Trainable params: 24,388,901
Non-trainable params: 0
__________________________________________________________________________________________________

```
