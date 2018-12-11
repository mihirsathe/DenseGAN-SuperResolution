# %load DenseSRGAN.py
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
import numpy as np
import DenseBlock as db # import functions to build dense blocks

class DenseSRGAN:
    
  def __init__(self, hr_img_size=(64,64,4), down_factor=4,
               num_layers_in_blk = 5, num_dense_blks=2,
               growth_rate=16, num_filters=64,
               dropout_rate=0.2, weight_decay=1e-4,weights_path=None):
    
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
    self.build_models()
    
    # TODO: Things (Not in order of importance)
    # 1. Load Data - Currently data is loaded outside then passed
    #    to the train function. Might be better to:
    #    a. Pass data to the constructor in order to get
    #       the input size before building the network
    # 2. Fix training function to do the useful things:
    #    a. Need to add additional loss functions 
    #       PSNR and maybe feature matching (ie VGG pretrain)
    #    b. like save a benchmark image on interval ie epoch
    #    c. Add a validation loss and early stopping
    # 3. Review network architecture from paper especially
    #    input/output layers ie input C3 and output FCN C1

  def train(self, datahr, datalr, epochs=1, batch_size=16, callbacks=None, save_interval=0):
    
    # Shuffle data indices    
    num_train = len(datahr)
    idx = np.random.permutation(list(range(num_train - 1)))
    # TODO: This just throws away the remaining expamples if not batch_size not divisor of num_train
    num_batches = num_train/batch_size if num_train%batch_size == 0 else num_train/batch_size - 1
    num_batches = int(num_batches)
    
    # if save_interval > 0: grab and hold a random lr input for benchmarking
    if save_interval > 0:
        bench_idx = np.random.randint(1,num_train - 1,1)
        
    for epoch in range(epochs):
        # Grab batch_size images from training data both lr and hr
        for batch_idx in range(num_batches):
            bix_begin = batch_idx
            bix_end   = batch_idx + batch_size
            
            # generate fake hr images with generator.predict(lr_imgs) size of batch
            batch_lr = datalr[bix_begin:bix_end,:,:,:]
            batch_hr = datahr[bix_begin:bix_end,:,:,:]
            fake_hr = self.G.predict(batch_lr)
            x_tr = np.concatenate((batch_hr,fake_hr))
            y_tr = np.ones([len(x_tr),1])
            y_tr[batch_size:] = 0 
            
            # Train the discriminator alone
            d_loss = self.DM.train_on_batch(x_tr,y_tr)
            
             # Grab another batch_size of images just for end to end training for adversarial model self.AM
            y_tr = np.ones([batch_size, 1])
            x_tr = batch_lr
            a_loss = self.AM.train_on_batch(x_tr, y_tr)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (batch_idx, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            
            # Print loss, call callbacks, save benchmarks if interval, etc...
 
        
        print(log_mesg)
        self.G.save("generator_weights.h5")
        
      
      
  def show_size(self):
    print("W: {0} / H: {1} / C: {2}".format(self.imhr_w,self.imhr_h,self.imhr_c))
  
  def get_summary(self):
        if self.D is not None:
            self.D.summary()
        if self.G is not None:
            self.G.summary()
    
  '''Discriminator Archicture
      Generate the discriminator model
  '''
  def init_discriminator(self):
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
    
    #if self.weights_path is not None:
    #  self.D.load_weights(self.weigths_path)
    
    return self.D
    
    
    
  '''Generator Archicture
      Generate the generator model
  '''
  def init_generator(self):
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
    
    if self.weights_path is not None:
      self.G.load_weights(self.weights_path)
    
    return self.G
  
    
    
  def build_models(self):
    
    lr_disc = 1e-4
    lr_adv = 1e-2
    decay = 6e-6
    
    # Initialize architectures
    self.init_discriminator()
    self.init_generator()
    
    # Build Discriminator Model
    optimizer = SGD(lr=lr_disc, decay=decay, nesterov=True)
    self.DM = Sequential()
    self.DM.add(self.D)
    self.DM.compile(loss='binary_crossentropy',
                             optimizer = optimizer,
                             metrics=['accuracy'])
    
    optimizer = ADAM(lr=lr_adv, amsgrad=True)
    self.D.trainable = False
    self.AM = Sequential()
    self.AM.add(self.G)
    self.AM.add(self.D)
    self.AM.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])