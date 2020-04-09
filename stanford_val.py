from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
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

log_name = './log/train_stanford_fcn8.log'
VGG_Weights_path = os.path.expanduser(os.path.join(
    '~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
#best_weight_path = "stanford_fcn8_weights.best.hdf5"
best_weight_path = "gan_gen545.best.hdf5"
model_path = './models/stanford_fcn8.h5'
hist_path = './hist/trainHistory_stanford_fcn8.txt'

siz = 224
input_height, input_width = siz, siz
output_height, output_width = siz, siz

n_classes = 12
batch_size = 32
epochs = 1000

sys.stdout = Logger(log_name, sys.stdout)

dir_testseg = "./dataset/dataset1/annotations_prepped_test"  # annotations_prepped_train
dir_testimg = "./dataset/dataset1/images_prepped_test"  # images_prepped_train
images = os.listdir(dir_testimg)
images.sort()
segmentations = os.listdir(dir_testseg)
segmentations.sort()

val_data,val_label = images,segmentations

X = []
Y = []
for im , seg in zip(val_data,val_label) :
    X.append( getImageArr(dir_testimg +"/"+ im , input_width , input_height )  )
    Y.append( getSegmentationArr( dir_testseg +"/"+ seg , n_classes , output_width , output_height )  )

x_val, y_val = np.array(X) , np.array(Y)


model = FCN8(VGG_Weights_path,n_classes)
model.load_weights(best_weight_path,by_name=True)


#Evaluate
#y_pred = model.generator.predict(x_val)
y_pred = model.predict(x_val)
y_predi = np.argmax(y_pred, axis=3)
y_vali = np.argmax(y_val, axis=3)
IoU(y_vali,y_predi)
visualize_prediction(x_val,y_predi,y_vali,n_classes)

