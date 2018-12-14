import tensorflow as tf
from tensorflow.data import Dataset
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import datetime
import DenseBlock as db # import functions to build dense blocks

class DenseSRGAN:
    
  def __init__(self, dir_pfx, datahr, datalr, gpu_list=None,
               hr_img_size=(64,64,4), down_factor=4,
               num_layers_in_blk = 5, num_dense_blks=2,
               growth_rate=16, num_filters=64,
               dropout_rate=0.2, weight_decay=1e-4,weights_path=None):
    
    # Batch Norm Epsillon
    self.eps = 1.1e-5
    
    self.dir_pfx           = dir_pfx

    self.num_gpus          = num_gpus
    
    self.datahr            = datahr
    self.datalr            = datalr
    
    hrshape = datahr.shape
    lrshape = datalr.shape
 
    self.imhr_w            = hrshape[1]
    self.imhr_h            = hrshape[2]
    self.im_c              = hrshape[3]
    self.imlr_w            = lrshape[1]
    self.imlr_h            = lrshape[2]
    self.down_factor       = int(self.imhr_w/self.imlr_w)
    
    # Square images
    assert self.imhr_w == self.imhr_h and self.imlr_w == self.imlr_h
    # Factor of 2
    assert self.down_factor%2 == 0
    # Same channels in lr and hr images
    assert hrshape[3] == lrshape[3]
 
    #self.down_factor       = down_factor
    #self.imhr_w            = hr_img_size[0]
    #self.imhr_h            = hr_img_size[1]
    #self.im_c              = hr_img_size[2]
    #self.down_factor       = down_factor
    # downsample power of 2 TODO: More Checks
    #assert down_factor%2 == 0
    #self.imlr_w            = self.imhr_w/down_factor
    #self.imlr_h            = self.imhr_h/down_factor
    
    self.num_dense_blks    = num_dense_blks
    self.blk_layers        = num_layers_in_blk
    self.growth_rate       = growth_rate
    self.num_filters       = num_filters
    self.dropout_rate      = dropout_rate
    self.weight_decay      = weight_decay
    self.weights_path      = weights_path
    self.disc              = None
    self.gen               = None
    self.disc_model        = None
    self.adv_model         = None
    
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
    
  def show_size(self):
    print("W: {0} / H: {1} / C: {2}".format(self.imhr_w,self.imhr_h,self.imhr_c))
  
  def get_summary(self):
        if self.D is not None:
            self.D.summary()
        if self.gen is not None:
            self.gen.summary()

            
  '''Discriminator Archicture
      Generate the discriminator model
  '''
  
  def init_discriminator(self):
    # If already defined return
    if self.disc:
      return self.disc
    
    # Defined the discriminator network
    hr_input = Input(shape=[self.imhr_w,self.imhr_h,self.im_c], name='highres_input')
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
    x = Conv2D(16, (1, 1),
               name=base_name + '_conv1', use_bias=False)(x)
    # Fix the dimension TODO: try to figure out the paper
    x = Dense(d_fmaps[0])(x)
    x = Activation('relu', name=base_name + '_relu2')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid', name=base_name + '_sigmoid')(x)
    
    return Model(hr_input, x, name='discriminator')
    
    
    
  '''Generator Archicture
      Generate the generator model
  '''
  def init_generator(self):
    # Inital number of feature maps
    num_filts = self.num_filters
    
    # If already defined return
    if self.gen:
      return self.gen
       
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
      # num_filts += self.growth_rate
      
      x = db.dense_block(x, blk_idx, self.blk_layers, # Grow maps here TODO: Try other way
                            num_filts, self.dropout_rate,
                            self.weight_decay)
      
      num_filts += self.growth_rate
      
      x = db.trans_layer(x, blk_idx, num_filts,
                            self.dropout_rate, self.weight_decay) # Maps same here
    
    # Final Dense Block (No Growth)
    #x = db.dense_block(x, self.num_dense_blks + 1, self.blk_layers, 
    #                      num_filts, self.dropout_rate,
    #                      self.weight_decay)
    
    # Batch Norm / Dropout / Max Pool / etc?? Paper S3.1 "C3" What??
    
    base_name = 'FC_out_'
    ## OUTPUT LAYER / REPLACE DENSE BLOCK ABOVE paper s 3.1    
    # TODO: Fully Convolutional Output See Ref (same for discriminator)    
    x = Conv2D(num_filts, (3,3), padding='same', name=base_name + '_c3x3s1', use_bias=False)(x)
    x = BatchNormalization(name=base_name + '_bn_1')(x)
    x = Activation('relu', name=base_name + '_relu_1')(x)
    x = Conv2D(num_filts, (1,1), padding='same', name=base_name + '_c1x1s1', use_bias=False)(x)
    x = BatchNormalization(name=base_name + '_bn_2')(x)
    x = Activation('relu', name=base_name + '_relu_2')(x)

    #x = BatchNormalization(epsilon=self.eps, name=base_name + '_bn')(x)
    #x = Activation('relu', name=base_name + '_relu')(x)
    x = Conv2D(self.im_c, (1, 1), name=base_name + '_conv1', use_bias=False)(x)
    x = Activation('sigmoid', name=base_name + '_sigmoid')(x)

    return Model(lr_input, x, name='generator')

  def build_models(self):
    
    lr    = 4e-4
    clip  = 1.0
    decay = 1e-8
    
    # Initialize architectures
    self.disc  = self.init_discriminator()
    self.gen = self.init_generator()
    
    if self.weights_path is not None:
      self.gen.load_weights(self.weights_path +
                            'generator_weights.h5')
      self.disc.load_weights(self.weights_path +
                             'discriminator_weights.h5')
    
    # Create the model 
    if self.num_gpus is not None:
      self.disc_model = multi_gpu_model(self.disc, gpus=self.num_gpus)
    else:
      self.disc_model = self.disc

    # Compile Discriminator Model
    #doptimizer = SGD(lr=lr, nesterov=True, clipvalue=clip)
    doptimizer = RMSprop(lr=lr, decay=decay, clipvalue=clip)
    self.disc_model.compile(loss='mse',
                      optimizer=doptimizer,
                      metrics=['accuracy'])    

    # Compile Adversarial Model
    #goptimizer = Adam(clipvalue=clip)
    goptimizer = RMSprop(lr=lr/2, decay=decay, clipvalue=clip)
    self.disc.trainable = False
    im_lr = Input(shape=(self.imlr_w,self.imlr_h,self.im_c))
    im_hr = Input(shape=(self.imhr_w,self.imhr_h,self.im_c))
    
    # Generated HR Images
    gen_hr = self.gen(im_lr)
    
    # Discriminator on Generator Output
    disc_gen_hr = self.disc(gen_hr)
    
    self.adv_model = Model(im_lr, disc_gen_hr)

    # Create the model 
    if self.num_gpus is not None:
      self.adv_model = multi_gpu_model(Model(im_lr, disc_gen_hr), gpus=self.num_gpus)
    else:
      self.adv_model = Model(im_lr, disc_gen_hr)    
    
    
    self.adv_model.compile(loss=['binary_crossentropy'],
                           #loss_weights=[1e-3,1],
                           optimizer=goptimizer,
                           metrics=['accuracy'])
    
    #self.adv_model = Sequential()
    #self.adv_model.add(self.gen)
    #self.adv_model.add(self.gen)
    #self.adv_model.compile(loss='binary_crossentropy',
    #                optimizer=goptimizer,
    #                metrics=['accuracy'])
    
    
  ''' TRAIN '''
  def train(self, datahr=None, datalr=None,
            epochs=1, batch_size=16, callbacks=None,
            save_interval=1, view_interval=1, 
            bench_idx=None, verbose=False):
     
    #datahr = self.datahr if datahr is None else datahr
    #datalr = self.datalr if datalr is None else datalr
    
            
    running_loss = []
    
    # TODO: This just throws away the remaining expamples if not batch_size not divisor of num_train
    num_train   = len(self.datahr)
    num_batches = num_train/batch_size if num_train%batch_size == 0 else num_train/batch_size - 1
    num_batches = int(num_batches)
    
    # if save_interval > 0: grab and hold a random lr input for benchmarking
    if save_interval > 0:

        if bench_idx is None:
            bench_idx = np.random.randint(1,num_train - 1,1)
        else:
            bench_idx = np.array([bench_idx,])

        bench_lr  = self.datalr[bench_idx,:,:,:]
        bench_hr  = self.datahr[bench_idx,:,:,:]
        
    for epoch in range(epochs):
        
        # Shuffle the indices TODO: Shuffle batch in place
        idx = np.random.permutation(list(range(num_train - 1)))
        epoch_start_time = datetime.datetime.now()
        # Grab batch_size images from training data both lr and hr
        for batch_idx in range(int(num_batches/2)): # Take 2 batches per round
            bix_begin = batch_idx*batch_size
            bix_end   = bix_begin+batch_size

            # generate fake hr images with generator.predict(lr_imgs) size of batch
            ti = datetime.datetime.now()
            if verbose: print('Start Shuffling data...')
            batch_lr = np.array([self.datalr[i,:,:,:] for i in idx[bix_begin:bix_end]])
            batch_hr = np.array([self.datahr[i,:,:,:] for i in idx[bix_begin:bix_end]])        
            if verbose: print('Done shuffling. Time: {0}'.format(datetime.datetime.now() - ti))
            
            if verbose: print('Making Predictions...')
            ti = datetime.datetime.now()
            x_gen = self.gen.predict(batch_lr)
            if verbose: print('Done predicting. Time: {0}'.format(datetime.datetime.now() - ti))

            x_tr  = batch_hr
            #x_tr  = np.concatenate((batch_hr,x_gen))
            y_tr  = np.ones((len(x_tr),) + (4,4,1))
            y_gen = np.zeros((len(x_gen),) + (4,4,1)) 
            
            if verbose: print('Training Discriminator...')
            ti = datetime.datetime.now()
            # Train the discriminator alone
            d_loss_hr  = self.disc_model.train_on_batch(x_tr, y_tr)
            d_loss_gen = self.disc_model.train_on_batch(x_gen, y_gen)
            if verbose: print('Done training. Time: {0}'.format(datetime.datetime.now() - ti))      
            
            # Grab another batch_size of images just for end to end training for adversarial model self.adv_model
            bix_begin = bix_end
            bix_end   = bix_begin + batch_size
            batch_lr = np.array([self.datalr[i,:,:,:] for i in idx[bix_begin:bix_end]])    
            
            y_tr = np.ones((batch_size,)+(4,4,1))
            x_tr = batch_lr
            
            #print('GAN Train Size: {0}'.format(x_tr.shape))

            if verbose: print('Training GAN...')
            ti = datetime.datetime.now()
            a_loss = self.adv_model.train_on_batch(x_tr, y_tr)
            if verbose: print('Done training GAN. Time: {0}'.format(datetime.datetime.now() - ti))   

            log_mesg = "%d:%d: [D loss_hr: %f, acc_hr: %f, loss_gen: %f, acc_gen: %f]" % \
                            (epoch, batch_idx, d_loss_hr[0], d_loss_hr[1], d_loss_gen[0], d_loss_gen[1])
            
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            
            # Print loss, call callbacks, save benchmarks if interval, etc...
            #print(log_mesg)
            running_loss.append([epoch] + [batch_idx] + list(d_loss_hr) + list(d_loss_gen) + list(a_loss))
 
        if epoch%view_interval == 0:
            print('Finished Epoch {0}... Time: {1}'.format(epoch, datetime.datetime.now()-epoch_start_time))
            print(log_mesg)



        # If save, save a pic
        if epoch%save_interval == 0:
            img = self.gen.predict(bench_lr).squeeze()
            #img = (img + 1)/2
            plt.ioff()
            plt.figure().suptitle('HR + Prediction: Epoch {0}'.format(epoch), fontsize=20)
            plt.subplot(1,2,1)
            plt.imshow(bench_hr.squeeze())
            plt.subplot(1,2,2)
            plt.imshow(img)
            plt.savefig('{0}images/bench_epoch_{1}'.format(self.dir_pfx,epoch))
        
        np.save(self.dir_pfx + 'loss_logging/loss_log.npy', arr=np.array(running_loss))
        
        if epoch%10 == 0:
            print('Saving weights at epoch {0}'.format(epoch))
            self.gen.save(self.dir_pfx + 'weights/generator_weights.h5')
            self.disc.save(self.dir_pfx + 'weights/discriminator_weights.h5')
