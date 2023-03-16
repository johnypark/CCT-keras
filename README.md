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
 layer_normalization_26 (LayerN  (None, 196, 384)    768         ['add_25[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 multi_head_self_attention_13 (  (None, None, 384)   591360      ['layer_normalization_26[0][0]'] 
 MultiHeadSelfAttention)                                                                          
                                                                                                  
 drop_path_26 (DropPath)        (None, None, 384)    0           ['multi_head_self_attention_13[0]
                                                                 [0]']                            
                                                                                                  
 add_26 (Add)                   (None, 196, 384)     0           ['add_25[0][0]',                 
                                                                  'drop_path_26[0][0]']           
                                                                                                  
 layer_normalization_27 (LayerN  (None, 196, 384)    768         ['add_26[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 feed_forward_network_13 (FeedF  (None, 196, 384)    886272      ['layer_normalization_27[0][0]'] 
 orwardNetwork)                                                                                   
                                                                                                  
 drop_path_27 (DropPath)        (None, 196, 384)     0           ['feed_forward_network_13[0][0]']
                                                                                                  
 add_27 (Add)                   (None, 196, 384)     0           ['add_26[0][0]',                 
                                                                  'drop_path_27[0][0]']           
                                                                                                  
 layer_normalization_28 (LayerN  (None, 196, 384)    768         ['add_27[0][0]']                 
 ormalization)                                                                                    
                                                                                                  
 dense (Dense)                  (None, 196, 1)       385         ['layer_normalization_28[0][0]'] 
                                                                                                  
 tf.linalg.matmul (TFOpLambda)  (None, 1, 384)       0           ['dense[0][0]',                  
                                                                  'layer_normalization_28[0][0]'] 
                                                                                                  
 flatten (Flatten)              (None, 384)          0           ['tf.linalg.matmul[0][0]']       
                                                                                                  
 dropout_1 (Dropout)            (None, 384)          0           ['flatten[0][0]']                
                                                                                                  
 dense_1 (Dense)                (None, 1000)         385000      ['dropout_1[0][0]']              
                                                                                                  
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

{'conv2d/kernel:0': (0, tf.float32, TensorShape([7, 7, 3, 192])),
 'conv2d_1/kernel:0': (1, tf.float32, TensorShape([7, 7, 192, 384])),
 'layer_normalization/gamma:0': (2, tf.float32, TensorShape([384])),
 'layer_normalization/beta:0': (3, tf.float32, TensorShape([384])),
 'multi_head_self_attention/dense_query/kernel:0': (4,
  tf.float32,
  TensorShape([384, 384])),
 'multi_head_self_attention/dense_query/bias:0': (5,
  tf.float32,
  TensorShape([384])),
 'multi_head_self_attention/dense_key/kernel:0': (6,
  tf.float32,
  TensorShape([384, 384])),
 'multi_head_self_attention/dense_key/bias:0': (7,
  tf.float32,
  TensorShape([384])),
 'multi_head_self_attention/dense_value/kernel:0': (8,
  tf.float32,
  TensorShape([384, 384])),
 'multi_head_self_attention/dense_value/bias:0': (9,
  tf.float32,
  TensorShape([384])),
 'multi_head_self_attention/dense_out/kernel:0': (10,
  tf.float32,
  TensorShape([384, 384])),
 'multi_head_self_attention/dense_out/bias:0': (11,
  tf.float32,
  TensorShape([384])),
 'layer_normalization_1/gamma:0': (12, tf.float32, TensorShape([384])),
 'layer_normalization_1/beta:0': (13, tf.float32, TensorShape([384])),
 'feed_forward_network/dense_hidden/kernel:0': (14,
  tf.float32,
  TensorShape([384, 1152])),
 'feed_forward_network/dense_hidden/bias:0': (15,
  tf.float32,
  TensorShape([1152])),
 'feed_forward_network/dense_out/kernel:0': (16,
  tf.float32,
  TensorShape([1152, 384])),
 'feed_forward_network/dense_out/bias:0': (17, tf.float32, TensorShape([384])),
 'layer_normalization_2/gamma:0': (18, tf.float32, TensorShape([384])),
 'layer_normalization_2/beta:0': (19, tf.float32, TensorShape([384])),

```


# Results

Results and weights are adpoted directly from the official PyTorch implementation (https://github.com/SHI-Labs/Compact-Transformers).
Type can be read in the format `L/PxC` where `L` is the number of transformer
layers, `P` is the patch/convolution size, and `C` (CCT only) is the number of
convolutional layers.

## CIFAR-10 and CIFAR-100

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td>
            <td><b>Pretraining</b></td> 
            <td><b>Epochs</b></td> 
            <td><b>PE</b></td>
	    <td><b>Source</b></td>
            <td><b>CIFAR-10</b></td> 
            <td><b>CIFAR-100</b></td> 
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=6>CCT-7/3x1</td>
            <td rowspan=6>None</td>
            <td rowspan=2>300</td>
            <td rowspan=2>Learnable</td>
	    <td>Official Pytorch</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth">96.53%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth">80.92%</a></td>
        </tr>
	<tr>    
	    <td>CCT-keras</td>
            <td> TBD </td>
            <td> TBD </td>
        </tr>
        <tr>
            <td rowspan=2>1500</td>
            <td rowspan=2>Sinusoidal</td>		
	    <td>Official Pytorch</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_1500epochs.pth">97.48%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_1500epochs.pth">82.72%</a></td>
        </tr>
	<tr>    
	    <td>CCT-keras</td>
            <td> TBD </td>
            <td> TBD </td>
        </tr>
        <tr>
            <td rowspan=2>5000</td>
            <td rowspan=2>Sinusoidal</td>			
	    <td>Official Pytorch</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth">98.00%</a></td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth">82.87%</a></td>
	</tr>
	<tr>    
	    <td>CCT-keras</td>
            <td> TBD</td>
            <td> TBD</td>
        </tr>
    </tbody>
</table>

## Flowers-102

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td>
            <td><b>Pre-training</b></td>
            <td><b>PE</b></td>
            <td><b>Image Size</b></td>
	    <td><b>Source</b></td>
            <td><b>Accuracy</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>CCT-7/7x2</td>
            <td rowspan=2>None</td>
            <td rowspan=2>Sinusoidal</td>
            <td rowspan=2>224x224</td>
	    <td>Official Pytorch</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth">97.19%</a></td>
        </tr>	    
	<tr>    
	    <td>CCT-keras</td>
            <td> TBD</td>
        </tr>
        <tr>
            <td rowspan=2>CCT-14/7x2</td>
            <td rowspan=2>ImageNet-1k</td>
            <td rowspan=2>Learnable</td>
            <td rowspan=2>384x384</td>
	    <td>Official Pytorch</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth">99.76%</a></td>
	<tr>    
	    <td>CCT-keras</td>
            <td> TBD</td>
        </tr>
        </tr>
    </tbody>
</table>

## ImageNet

<table style="width:100%">
    <thead>
        <tr>
            <td><b>Model</b></td> 
            <td><b>Type</b></td> 
            <td><b>Resolution</b></td> 
            <td><b>Epochs</b></td> 
	    <td><b># Params</b></td> 
            <td><b>MACs</b></td>
            <td><b>Top-1 Accuracy</b></td>            
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1><a href="https://github.com/google-research/vision_transformer/">ViT</a></td>
            <td>12/16</td>
	        <td>384</td>
	        <td>300</td>
            <td>86.8M</td>
            <td>17.6G</td>
            <td>77.91%</td>
        </tr>
        <tr>
            <td rowspan=2>CCT</td>
            <td>14/7x2</td>
	        <td>224</td>
            <td>310</td>
            <td>22.36M</td>
            <td>5.11G</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth">80.67%</a></td>
        </tr>
        <tr>
            <td>14/7x2</td>
	        <td>384</td>
            <td>310 + 30</td>
            <td>22.51M</td>
            <td>15.02G</td>
            <td><a href="https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth">82.71%</a></td>
        </tr>
    </tbody>
</table>
