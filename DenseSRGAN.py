import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
import DenseBlock as db # import functions to build dense blocks

class DenseSRGAN:
	
  def __init__(self,hr_img_size=(64,64,4), down_factor=4,
               num_layers_in_blk = 5, num_dense_blks=2,
               growth_rate=16, num_filters=64,
               dropout_rate=0.0, weight_decay=1e-4,weights_path=None):
    
    # Batch Norm Epsillon
    self.eps = 1.1e-5
    
    self.num_dense_blks = num_dense_blks
    self.imhr_w            = hr_img_size[0]
    self.imhr_h            = hr_img_size[1]
    self.im_c              = hr_img_size[2]
    self.down_factor       = down_factor
    
    # downsample power of 2 TODO: More Checks
    assert down_factor%2 == 0
    
    self.imlr_w            = self.imhr_w/down_factor
    self.imlr_h            = self.imhr_h/down_factor
    
    self.blk_layers        = num_layers_in_blk
    self.growth_rate       = growth_rate
    self.num_filters       = num_filters
    self.dropout_rate      = dropout_rate
    self.weight_decay      = weight_decay
    self.weights_path      = weights_path
    self.D                 = None
    self.G                 = None
    self.AM                = None
    self.DM                = None
    
    # Initialize
    self.discriminator()
    self.generator()
    self.discriminator_model()
    self.adversarial_model()
    
    
    # TODO: Things
    # 1. Load Data - Grab a DataLoader object if possible
    #    Create a local instance of loader to use in training

  def train(self, epochs=10, batch_size=16, callbacks=None, save_interval=0):
    # if save_interval > 0: grab and hold a random lr input for benchmarking
    
    # Grab batch_size images from training data both lr and hr
    
    # generate fake hr images with generator.predict(lr_imgs) size of batch
    
    # concat real training imgs and fake gen images for discriminator training and labels
    
    # get discriminator loss and train_on_batch(x,y)
    
    # Grab another batch_size of images just for end to end training for adversarial model self.AM
    
    # Print loss, call callbacks, save benchmarks if interval, etc...
    print('training....')
     
      
      
  def show_size(self):
    print("W: {0} / H: {1} / C: {2}".format(self.imhr_w,self.imhr_h,self.imhr_c))
    
  '''Discriminator Archicture
      Generate the discriminator model
  '''
  def discriminator(self):
    # If already defined return
    if self.D:
      return self.D
    
    # Defined the discriminator network
    hr_input = Input(shape=(self.imhr_w,self.imhr_h,self.im_c), name='highres_input')
    # Number of feature maps in each layer 3.4
    d_fmaps = [64, 128, 256, 512] 
    
    x = hr_input
    # Four layers with C3x3s2 
    for i,fmaps in enumerate(d_fmaps):
      base_name = 'l_{0}'.format(i)
      x = Conv2D(fmaps, (3,3),
                 padding='same', strides=(2,2),
                 name=base_name + '_conv', use_bias=False)(x)
      x = BatchNormalization(name=base_name + '_bn')(x)
      x = Activation('relu', name=base_name + '_relu')(x)
      
    # TODO: Fully Convolutional Output See Ref (same for discriminator) 3.4
    base_name = 'FC_out_' + str(i + 1)
    
    x = BatchNormalization(epsilon=self.eps, name=base_name + '_bn')(x)
    x = Activation('relu', name=base_name + '_relu')(x)
    x = Conv2D(self.im_c, (1, 1), name=base_name + '_conv1', use_bias=False)(x)
    # Fix the dimension TODO: try to figure out the paper
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid', name=base_name + '_sigmoid')(x)
    
    self.D = Model(hr_input, x, name='discriminator')
    #self.D.summary()
    
    if self.weights_path is not None:
      self.D.load_weights(self.weigths_path)
    
    return self.D
    
    
    
  '''Generator Archicture
      Generate the generator model
  '''
  def generator(self):
    # Inital number of feature maps
    num_filts = self.num_filters
    
    # If already defined return
    if self.G:
      return self.G
       
    # Define the generator network
    lr_input = Input(shape=(self.imlr_w,self.imlr_h,self.im_c), name='lowres_input')
    
    # Initial Convolution
    x = Conv2D(num_filts, (3,3), padding='same', name='init_conv', use_bias=False)(lr_input)
    # Batch Norm / Dropout / Max Pool / etc?? Paper S3.1 "C3" What??
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)
    
    # TODO: Growth rate happens in dense block whereas in DenseNet growth 
    # occurs in transistion block. Paper 3.1 says growth in Dense Block. DONE
    
    # Dense Blocks
    for blk_idx in range(1, self.num_dense_blks + 1): # Consider 1st and last layer
      # TODO: Number of layers/block is constant blk_layers, can change per
      # block blk_layers[i] if param is list
      num_filts += self.growth_rate
      
      x = db.dense_block(x, blk_idx, self.blk_layers, # Grow maps here
                            num_filts, self.dropout_rate,
                            self.weight_decay)
      
      x = db.trans_layer(x, blk_idx, num_filts,
                            self.dropout_rate, self.weight_decay) # Maps same here
      
    # Final Dense Block (No Growth)
    x = db.dense_block(x, self.num_dense_blks + 1, self.blk_layers, 
                          num_filts, self.dropout_rate,
                          self.weight_decay)
    
    # TODO: Fully Convolutional Output See Ref (same for discriminator)
    base_name = 'FC_out_'
    x = BatchNormalization(epsilon=self.eps, name=base_name + '_bn')(x)
    x = Activation('relu', name=base_name + '_relu')(x)
    x = Conv2D(self.im_c, (1, 1), name=base_name + '_conv1', use_bias=False)(x)
    x = Activation('sigmoid', name=base_name + '_sigmoid')(x)
    
    self.G = Model(lr_input, x, name='generator')
    self.G.summary()
    
    if self.weights_path is not None:
      self.G.load_weights(self.weigths_path)
    
    return self.G
  
  '''Compile Discriminator Archicture
      TODO: it
  '''
  def discriminator_model(self):
      if self.DM:
          return self.DM
      optimizer = RMSprop(lr=0.0002, decay=6e-8)
      self.DM = Sequential()
      self.DM.add(self.discriminator())
      self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
          metrics=['accuracy'])
      return self.DM
      
  '''Compile Generator Archicture
      TODO: it
  '''
  def adversarial_model(self):
      if self.AM:
          return self.AM
      optimizer = RMSprop(lr=0.0001, decay=3e-8)
      self.AM = Sequential()
      self.AM.add(self.G)
      self.AM.add(self.D)
      self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
          metrics=['accuracy'])
      return self.AM
    