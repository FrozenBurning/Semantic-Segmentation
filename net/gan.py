'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-25 20:04:28
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-04-08 19:31:02
'''
from net.fcn8 import *
from keras.models import Sequential
from keras.layers import Reshape
from keras import backend as K
import six
from utils.visualize import IoU
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

class GAN():
    def __init__(self, VGG_Weights_path, opt,n_classes=21, input_width=224, input_height=224):
        self.n_classes = n_classes
        self.input_width = input_width
        self.input_height = input_height
        self.vgg_weights_path = VGG_Weights_path
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()

        #self.discriminator.compile(loss='mse',optimizer=opt,metrics=['accuracy'])
        self.discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

        img_input = Input(shape=(self.input_height, self.input_width, 3))
        label =self.generator(img_input)
        self.discriminator.trainable = False
        validity =self.discriminator(label)
        self.combined = Model(img_input,validity)
        #self.combined.compile(loss='mse',optimizer=opt)
        self.combined.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    def _build_generator(self):
        model = FCN8(VGG_Weights_path=self.vgg_weights_path, nClasses=self.n_classes,
                     input_height=self.input_height, input_width=self.input_width)
        #model.load_weights('./small_fcn8_weights.best.hdf5',by_name=True)
        return model

    def _build_discriminator(self):
        img_input = Input(
            shape=(self.input_height, self.input_width, self.n_classes))
        IMAGE_ORDERING = "channels_last"
        def d_block(layer_input,filters,strides=1,bn=True):
            d = Conv2D(filters,(3,3),strides=strides)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        df = 64
        d1 = d_block(img_input,df,bn=False)
        d2 = d_block(d1,df,strides=2)
        d3 = d_block(d2,df*2)
        d4 = d_block(d3,df*2,strides=2)
        d5 = d_block(d4,df*4)
        d6 = d_block(d5,df*4,strides=2)
        d7 = d_block(d6,df*8)
        d8 = d_block(d7,df*8,strides=2)

        d9 =Dense(df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)


        # Block1
        x = Conv2D(96, (3, 3), activation='relu',
                   name='block1_conv1')(img_input)
        x = Conv2D(128, (3, 3), activation='relu',
                   name='block1_conv2')(x)
        x = Conv2D(128, (3, 3), activation='relu',
                   name='block1_conv3')(x)
        x = MaxPool2D((2, 2), strides=2, name='block1_pool')(x)

        # Block2
        x = Conv2D(256, (3, 3), activation='relu',
                   name='block2_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   name='block2_conv2')(x)
        x = MaxPool2D((2, 2), strides=2, name='block2_pool')(x)

        # Block3
        x = Conv2D(256, (3, 3), activation='relu',
                   name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   name='block3_conv2')(x)
        
        #x = Flatten()(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        
        d10 = Flatten()(d10)
        d10 = Dense(256,kernel_regularizer=l2(1e-3))(d10)
        d10 = LeakyReLU(0.2)(d10)
        validity = Dense(1,activation='sigmoid')(d10)        
        model = Model(img_input, validity)
        model.summary()
        
        return model

    def _onehot_encode(self,label_map,batch_size):
        label_idx = np.argmax(label_map,axis=3)
        onehot = np.zeros((batch_size,self.input_height,self.input_width,self.n_classes))
        img = label_idx
        for c in range(self.n_classes):
            onehot[:,:,:,c] = (img == c).astype(int)
        
        return K.cast_to_floatx(onehot)


    def train(self,x_train,y_train,x_val=None,y_val=None,epochs = 100,batch_size = 32,interval = 1):
        #valid = np.ones((batch_size,self.input_height,self.input_width,1))
        #fake = np.zeros((batch_size,self.input_height,self.input_width,1))
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        #valid = np.ones((batch_size,11,11,1))
        #fake = np.zeros((batch_size,11,11,1))
        index =  np.arange(len(x_train))
        steps = len(x_train)//batch_size
        maxiou = 0
        for epoch in range(epochs):
            np.random.shuffle(index)
            if epoch % interval == 0:
                y_pred = self.generator.predict(x_val)
                y_predi = np.argmax(y_pred,axis=3)
                y_vali = np.argmax(y_val,axis=3)
                tmp = IoU(y_vali,y_predi)
                if maxiou < tmp:
                    maxiou = tmp
                    self._check_point()

            for batch in range(steps):
                # idx = np.random.randint(0,x_train.shape[0],batch_size)
                idx = index[batch*batch_size:(batch+1)*batch_size]
                imgs = x_train[idx]
                real_label = y_train[idx]    
                # on batch
                gen_labels = self.generator.predict(imgs)
                gen_labels = self._onehot_encode(gen_labels,batch_size)            
                #print(gen_labels.shape)
                #print(gen_labels[0,0,0,:])
                #print(real_label.shape)
                #print(real_label[0,0,0,:])

                d_loss_real = self.discriminator.train_on_batch(real_label,valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_labels,fake)
                d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

                g_loss = self.combined.train_on_batch(imgs,valid)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0],100*g_loss[1]))


    def save_model(self,path):
        self.generator.save(path)
        self.discriminator.save('discrim_'+path)

    def _check_point(self):
        print('check point!')
        self.generator.save('./gan_gen.best.hdf5')
        self.discriminator.save('./gan_discrim.best.hdf5')
    
        

