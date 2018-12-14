import tensorflow as tf
from keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization


def dense_comp_fn(x, idx, layer_idx,
                  num_filters, dropout_rate=None,
                  weight_decay=1e-4):
  ''' Composite function H_l of dense block see https://arxiv.org/abs/1608.06993
    BN+ReLU+C1x1s1 (Bottleneck) + BN+ReLU+C3x3s1 (Basic Unit Sequence) 

      # Arguments
          x: Input 
          idx: Dense Block Index
          layer_idx: layer index within each dense block H_l
          num_filters: number of filters for output
          dropout_rate: dropout rate
          weight_decay: weight decay factor
  '''
  eps = 1.1e-5
  name_base = 'H' + str(idx) + '_' + str(layer_idx)
  
  # 1x1 Convolution (Bottleneck layer)
  feat_maps = num_filters * 4 # 4 from Densenet paper bottleneck
  # BN+ReLU+C1x1s1 (Bottleneck)
  x = BatchNormalization(epsilon=eps, name=name_base + '_btlnck_bn')(x)
  #x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
  x = Activation('relu', name=name_base + '_btlnck_relu')(x)
  x = Conv2D(feat_maps, (1, 1), name=name_base + '_btlnck_conv', use_bias=False)(x)
  if dropout_rate:
      x = Dropout(dropout_rate)(x)

  # BN+ReLU+C3x3s1 (Basic Unit Sequence) 
  x = BatchNormalization(epsilon=eps, name=name_base + '_conv_bn')(x)
  #x = Scale(axis=concat_axis, name=name_base + '_x2_scale')(x)
  x = Activation('relu', name=name_base + '_conv_relu')(x)
  x = Conv2D(num_filters, (3, 3), padding='same',
             name=name_base + '_conv_conv', use_bias=False)(x)

  if dropout_rate:
      x = Dropout(dropout_rate)(x)

  return x  

def trans_layer(x, idx, num_filter,
                      dropout_rate=None, weight_decay=1E-4):
  ''' Transpose Convolution 3x3s2 upsample for SR application
      # Arguments
          x: input tensor
          idx: index for dense block
          num_filter: number of filters
          weight_decay: weight decay factor
  '''
  trans_name_base = 'trans' + str(idx)

  x = Conv2DTranspose(num_filter, (3,3), strides=2, padding="same",
            name=trans_name_base, use_bias=False)(x)
  return x


def dense_block(x, idx, num_layers, num_filters,
                dropout_rate=None, weight_decay=1e-4):

  ''' Dense Block with num_layers and num_filters from DenseNet paper
      # Arguments
        x: input tensor
        idx: index of dense block in deep Dense
        num_layers: the number of layers of conv_block to append to the model.
        num_filters: number of feature maps at output
        dropout_rate: dropout rate
        weight_decay: weight decay factor
  '''
  eps = 1.1e-5

  for i in range(num_layers):
      layer = i + 1

      x1 = dense_comp_fn(x, idx, layer,
                        num_filters, dropout_rate,
                        weight_decay)

      lname = 'concat_' + str(idx) + '_' + str(layer)

      x = Concatenate(name=lname)([x, x1])
  
  return x