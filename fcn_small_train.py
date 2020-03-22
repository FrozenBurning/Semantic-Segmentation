# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Fully Convolutional Networks for semantic segmentation
# 
# In an image for the semantic segmentation, each pixcel is labeled with the class of its enclosing object. The semantic segmentation problem requires to make a classification at every pixel.
# 
# First, download data from:
# 
# https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view
# 
# and save the downloaded data1 folder in the current directory.

# %%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sns.set_style("whitegrid", {'axes.grid' : False})

# enter your path here
dir_seg = "./dataset/dataset1/annotations_prepped_train"  # annotations_prepped_train
dir_img = "./dataset/dataset1/images_prepped_train"  # images_prepped_train

ldseg = np.array(os.listdir(dir_seg))

## pick the first image file
fnm = ldseg[0]
print(fnm)

## read in the original image and segmentation labels
## Read first image from annotations_prepped_train and images_prepped_train with path "dir_seg +"/"+ fnm"

# my code
seg =  cv2.imread(dir_seg+'/'+fnm)  # image from annotations_prepped_train (360, 480, 3)
img_is = cv2.imread(dir_img+'/'+fnm) # image from images_prepped_train
print("seg.shape={}, img_is.shape={}".format(seg.shape,img_is.shape))


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
output_height , output_width = 224 , 224


ldseg = np.array(os.listdir(dir_seg))
# for fnm in ldseg[np.random.choice(len(ldseg),3,replace=False)]:
#     # randomly select on the training image
#     fnm = fnm.split(".")[0]
#     seg = cv2.imread(dir_seg  +"/"+ fnm + ".png") # (360, 480, 3)
#     img_is = cv2.imread(dir_img  +"/"+ fnm + ".png")
#     # assign color to its annotations_prepped_train image
#     seg_img = give_color_to_seg_img(seg,n_classes)

#     fig = plt.figure(figsize=(20,40))
#     ax = fig.add_subplot(1,4,1)
#     ax.imshow(seg_img)
    
#     ax = fig.add_subplot(1,4,2)
#     ax.imshow(img_is/255.0)
#     ax.set_title("original image {}".format(img_is.shape[:2]))
    
#     ax = fig.add_subplot(1,4,3)
#     ax.imshow(cv2.resize(seg_img,(input_height , input_width)))
    
#     ax = fig.add_subplot(1,4,4)
#     ax.imshow(cv2.resize(img_is,(output_height , output_width))/255.0)
#     ax.set_title("resized to {}".format((output_height , output_width)))
#     plt.show()

# %% [markdown]
# To simplify the problem, I will reshape all the images to the same size: (224,224). 
# 
# Since this is the iamge shape used in VGG and FCN model in this blog uses a network that takes advantage of VGG structure. The FCN model becomes easier to explain when the image shape is (224,224).

# %%
def getImageArr( path , width , height ):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
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


images = os.listdir(dir_img)
images.sort()
segmentations  = os.listdir(dir_seg)
segmentations.sort()
    
X = []
Y = []
for im , seg in zip(images,segmentations) :
    X.append( getImageArr(dir_img +"/"+ im , input_width , input_height )  )
    Y.append( getSegmentationArr( dir_seg +"/"+ seg , 12 , output_width , output_height )  )

X, Y = np.array(X) , np.array(Y)
print(X.shape,Y.shape)

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

# %% [markdown]
# # From classifier to dense FCN
# The recent successful deep learning models such as VGG are originally designed for classification task. The network stacks convolution layers together with down-sampling layers, such as max-pooling, and then finally stacks fully connected layers. Appending a fully connected layer enables the network to learn something using global information where the spatial arrangement of the input falls away.
# 
# # Fully convosutional network
# For the segmentation task, however, spatial infomation should be stored to make a pixcel-wise classification. FCN allows this by making all the layers of VGG to convolusional layers.
# 
# Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers usually found at the end of the network. Fully Convolutional Networks for Semantic Segmentation motivates the use of fully convolutional networks by "convolutionalizing" popular CNN architectures e.g. VGG can also be viewed as FCN.
# 
# The model used is FCN32 from Fully Convolutional Networks for Semantic Segmentation. It deplicates VGG16 net by discarding the final classifier layer and convert all fully connected layers to convolutions. Fully Convolutional Networks for Semantic Segmentation appends a 1 x 1 convolution with channel dimension the same as the number of segmentation classes (in our case, this is 12) to predict scores at each of the coarse output locations, followed by upsampling deconvolution layers which brings back low resolution image to the output image size. In our example, output image size is resized to (output_height, output_width) = (320, 320).
# 
# # Upsampling
# The upsampling layer brings low resolution image to high resolution. There are various upsamping methods. This presentation gives a good overview. For example, one may double the image resolution by duplicating each pixcel twice. This is so-called nearest neighbor approach and implemented in UpSampling2D. Another method may be bilinear upsampling, which linearly interpolates the nearest four inputs.
# 
# For the upsampling layer, See details at Fully Convolutional Networks for Semantic Segmentation.
# 
# Downloaded VGG16 weights from fchollet's Github
# This is a massive .h5 file (57MB).

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

sys.stdout = Logger('train.log', sys.stdout)
# %%
def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=12):
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
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='relu', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))


    model.load_weights(weights_path, by_name=True)
    return model
weight_decay = 1e-4/2
input_shape = (None, None, 3)
batchnorm_momentum = 0.95
classes = 12
#model = FCN_Vgg16_32s(weight_decay=weight_decay,input_shape=input_shape,batch_momentum=batchnorm_momentum,classes=classes)
model = FCN_Vgg16_32s(input_shape=input_shape)
model.summary()

# %% [markdown]
# Split between training and testing data

# %%
from sklearn.utils import shuffle
train_rate = 0.85
index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
index_test  = list(set(range(X.shape[0])) - set(index_train))
                            
X, Y = shuffle(X,Y)
X_train, y_train = X[index_train],Y[index_train]
X_test, y_test = X[index_test],Y[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# %%
from keras import optimizers
# Training data
#lr_base = 0.01 * (float(batch_size) / 16)
lr_base = 0.01 * (float(32) / 16)
#sgd = optimizers.SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
sgd = optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#print(X_train[0])
#print(y_train[0])
hist1 = model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=32,epochs=300,verbose=2)

model.save('./activ_relu.h5')
# %%

import pickle
 
with open('trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(hist1.history, file_pi)
