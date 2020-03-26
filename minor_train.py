'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 15:31:04
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-25 22:36:24
'''
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils.dataloader import *
from utils.logger import *
from net.gan import *
from net.fcn8 import *
from utils.visualize import *

log_name = './log/train_gan.log'
VGG_Weights_path = os.path.expanduser(os.path.join(
    '~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
best_weight_path = "ganweights.best.hdf5"
model_path = './models/gan_fcn8gen.h5'
hist_path = './hist/trainHistory_gan.txt'

siz = 224
input_height, input_width = siz, siz
output_height, output_width = siz, siz

n_classes = 12
batch_size = 32
epochs = 1000

sys.stdout = Logger(log_name, sys.stdout)

dir_seg = "./dataset/dataset1/annotations_prepped_train"  # annotations_prepped_train
dir_img = "./dataset/dataset1/images_prepped_train"  # images_prepped_train
images = os.listdir(dir_img)
images.sort()
segmentations = os.listdir(dir_seg)
segmentations.sort()

train_data,train_label,val_data,val_label = divider(images,segmentations)


X = []
Y = []
for im , seg in zip(train_data,train_label) :
    X.append( getImageArr(dir_img +"/"+ im , input_width , input_height )  )
    Y.append( getSegmentationArr( dir_seg +"/"+ seg , n_classes , output_width , output_height )  )

x_train, y_train = np.array(X) , np.array(Y)

X = []
Y = []
for im , seg in zip(val_data,val_label) :
    X.append( getImageArr(dir_img +"/"+ im , input_width , input_height )  )
    Y.append( getSegmentationArr( dir_seg +"/"+ seg , n_classes , output_width , output_height )  )

x_val, y_val = np.array(X) , np.array(Y)


adam = optimizers.Adam(1e-5,0.5)
#model = FCN8(VGG_Weights_path,n_classes)
#model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

#checkpoint = ModelCheckpoint(best_weight_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
#callbacks_list = [checkpoint]
#hist1 = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=batch_size,epochs=100,verbose=2,callbacks=callbacks_list)
#model.save(model_path)

#historian(hist1.history,hist_path)

gan_model = GAN(VGG_Weights_path, adam, n_classes,input_height=input_height, input_width=input_width)
gan_model.generator.load_weights(model_path,by_name=True)
gan_model.train(x_train, y_train, x_val,y_val,epochs=epochs, batch_size=batch_size)

#Evaluate
y_pred = gan_model.generator.predict(x_val)
y_predi = np.argmax(y_pred, axis=3)
y_vali = np.argmax(y_val, axis=3)
IoU(y_vali,y_predi)
visualize_prediction(x_val,y_predi,y_vali,n_classes)

gan_model.save_model(model_path)

