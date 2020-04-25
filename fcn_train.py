'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 15:31:04
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-14 15:41:40
'''
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import *
from utils.logger import *
from net.fcn8 import *
from net.deeplabv3 import *

log_name = './log/train_deeplab.log'
VGG_Weights_path = os.path.expanduser(os.path.join('~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
best_weight_path="./deeplabweights.best.hdf5"
model_path = './models/deeplab.h5'
hist_path='./hist/trainHistory_deeplab.txt'


siz = 224
input_height , input_width = siz,siz
output_height , output_width = siz,siz

sys.stdout = Logger(log_name, sys.stdout)


weight_decay = 1e-5
input_shape = (siz, siz, 3)
batchnorm_momentum = 0.95
classes = 21
BS = 16

#model = FCN8(VGG_Weights_path,nClasses = classes,input_height = input_height,input_width  = input_width)
model = Deeplabv3(input_shape=input_shape,classes = classes,activation='softmax')
#model.summary()

from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras import optimizers

#X_train,y_train,X_test,y_test = divider(X,y)
X_train,y_train=read_data_path(extension='/list/train_aug.txt')
#X_train,y_train = path2data(X,y)

print(len(X_train), len(y_train))
X_test,y_test = read_data_path(extension='/list/val.txt')
XX_test,yy_test = path2data(X_test,y_test,input_width,input_height,output_width,output_height)

print(XX_test.shape,yy_test.shape)

# Training data
model.load_weights(best_weight_path,by_name=True)
#sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
training_generator=DataGenerator(X_train,y_train,batch_size=BS,input_shape=[input_width,input_height],output_shape=[output_width,output_height])

checkpoint = ModelCheckpoint('deeplabminibatch.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
hist1 = model.fit_generator(training_generator,validation_data=(XX_test,yy_test),steps_per_epoch=len(X_train)//BS+1,epochs=100,verbose=2,use_multiprocessing=True,callbacks=callbacks_list)
model.save(model_path)

historian(hist1.history,hist_path)
