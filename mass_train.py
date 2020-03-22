

# %%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


sns.set_style("whitegrid", {'axes.grid' : False})

# enter your path here
#dir_seg = "./dataset1/annotations_prepped_train"  # annotations_prepped_train
#dir_img = "./dataset1/images_prepped_train"  # images_prepped_train
# dir_seg = './pascal_voc_seg/VOCdevkit'
# dir_img = './pascal_voc_seg/VOCdevkit'

def read_images(root='/home/throne/data/VOCdevkit/VOC2012', train=True):
    txt_fname = root + '/list/trainval_aug.txt'
    with open(txt_fname, 'r') as f:
        images = f.read().splitlines()
    print(len(images))
    data = [root+i.split()[0] for i in images]
    label = [root+i.split()[1] for i in images]
    return data, label

data,label=read_images()
print(len(data))
# %% [markdown]
# From the first section, we can see there are 12 segmentation classes and the image is from a driving car.
# 
# Assign color to annotations_prepped_train image

# %%
import random
def give_color_to_seg_img(seg,n_classes):
    '''
    seg : size is (input_width,input_height,3)
    assign color to each class 
    '''
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

input_height , input_width = 224,224
output_height , output_width = 224,224



# %% [markdown]
# To simplify the problem, I will reshape all the images to the same size: (224,224). 
# 
# Since this is the iamge shape used in VGG and FCN model in this blog uses a network that takes advantage of VGG structure. The FCN model becomes easier to explain when the image shape is (224,224).

# %%
def getImageArr( path , width , height ):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, ( width , height ))) /127.5-1
        return img

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels


# images = os.listdir(dir_img)
# images.sort()
# segmentations  = os.listdir(dir_seg)
# segmentations.sort()
images = data
segmentations = label
    
X = data
Y = label
#for im , seg in zip(images,segmentations) :
 #   X.append( getImageArr(im , input_width , input_height )  )
  #  Y.append( getSegmentationArr(seg , 21 , output_width , output_height )  )

#X, Y = np.array(X) , np.array(Y)
#print(X.shape,Y.shape)

# %% [markdown]
# Import Keras and Tensorflow to develop deep learning FCN models

# %%
## Import usual libraries
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd 
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
# from keras_contrib.applications import densenet
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras_applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
###############################################
# set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
# check python, keras, and tensorflow version
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))



# %%
# location of VGG weights
#the following code downloads the pretrained vgg16 weight.
def get_weights_path_vgg16():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    return weights_path
print( get_weights_path_vgg16())


# %%
def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
            #X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] , original_shape[2] , None))
            #X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)
        
class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            
            if input_shape[1] is not None:
                width = int(self.size[0] * input_shape[1])
            else:
                width = None
                            
            if input_shape[2] is not None:
                height = int(self.size[1] * input_shape[2])    
            else:
                height = None
                
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.filen = filename

    def write(self, message):
	    self.terminal.write(message)
	    with open(self.filen,'a') as f:
             f.write(message)

    def flush(self):
	    pass

sys.stdout = Logger('train_voc.log', sys.stdout)
# %%
VGG_Weights_path = os.path.expanduser(os.path.join('~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
def FCN8( nClasses ,  input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)

    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)   
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    
    vgg  = Model(  img_input , pool5  )
    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)

    return model

model = FCN8(nClasses     = 21,  
             input_height = 224, 
             input_width  = 224)
model.summary()

def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))


    model.load_weights(weights_path, by_name=True)
    return model
weight_decay = 1e-5
input_shape = (None, None, 3)
batchnorm_momentum = 0.95
classes = 21
#model = FCN_Vgg16_32s(weight_decay=weight_decay,input_shape=input_shape,batch_momentum=batchnorm_momentum,classes=classes)
#model = FCN_Vgg16_32s(input_shape=input_shape,classes=21)
#model.summary()

# %% [markdown]
# Split between training and testing data
import keras
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, datas, labels,batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):

        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_labels = [self.labels[k] for k in batch_indexs]
        X, y = self.data_generation(batch_datas,batch_labels)
        #print('get item!')
        
        return X,y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas,batch_labels):
        images = []
        labels = []

        for im ,seg in zip(batch_datas,batch_labels) :
            images.append( getImageArr(im, input_width , input_height )  )
            labels.append( getSegmentationArr(seg, 21 , output_width , output_height )  )
       
        return np.array(images), np.array(labels)

def batch_generator(datas,labels):
    images = []
    labels = []

    for im ,seg in zip(datas,labels) :
        images.append( getImageArr(im, input_width , input_height )  )
        labels.append( getSegmentationArr(seg, 21 , output_width , output_height )  )

    return np.array(images), np.array(labels)


# %%
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
train_rate = 0.85
index_train = np.random.choice(len(X),int(len(X)*train_rate),replace=False)
index_test  = list(set(range(len(X))) - set(index_train))
print(index_train)                            
#X, Y = shuffle(X,Y)
X_train, y_train = np.array(X)[index_train],np.array(Y)[index_train]
X_test, y_test = np.array(X)[index_test],np.array(Y)[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train[0])
XX_test=[]
yy_test=[]
for im , seg in zip(X_test,y_test) :
    XX_test.append( getImageArr(im , input_width , input_height )  )
    yy_test.append( getSegmentationArr(seg , 21 , output_width , output_height )  )
XX_test=np.array(XX_test)
yy_test=np.array(yy_test)

print(XX_test.shape,yy_test.shape)
# %%
from keras import optimizers
# Training data
#lr_base = 0.01 * (float(batch_size) / 16)
lr_base = 0.01 * (float(32) / 16)
sgd = optimizers.SGD(lr=1e-2, decay=5e-4, momentum=0.9, nesterov=True)
#sgd = optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
training_generator=DataGenerator(X_train,y_train)
#print(X_train[0])
#print(y_train[0])
#hist1 = model.fit(X_train,y_train,
 #                 validation_data=(X_test,y_test),
  #                batch_size=32,epochs=300,verbose=2)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]
hist1 = model.fit_generator(training_generator,validation_data=(XX_test,yy_test),steps_per_epoch=319,epochs=100,verbose=2,use_multiprocessing=True,callbacks=callbacks_list)
model.save('./augvoc.h5')
# %%

import pickle
 
with open('trainHistory_voc.txt', 'wb') as file_pi:
    pickle.dump(hist1.history, file_pi)
