'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 16:07:32
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-14 16:07:33
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils.dataloader import *
from utils.logger import Logger
from net.fcn8 import *
from net.fcn32 import *
from utils.visualize import *
from net.deeplab import*

log_name = './log/train_deeplab.log'
VGG_Weights_path = os.path.expanduser(os.path.join('~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
best_weight_path="deeplabweights.best.hdf5"
model_path = './models/deeplab.h5'
hist_path='./hist/trainHistory_deeplab.txt'

data,label = read_data_path(extension='/list/val.txt')
data = data[:720]
label = label[:720]
classes = 21
input_width,input_height=512,512
input_shape = (512,512,3)
#model = FCN8(VGG_Weights_path,classes,input_width,input_height)
model = Deeplabv3(input_shape=input_shape,classes = classes)

#model.load_weights(best_weight_path,by_name=True)
model.summary()

X_test,y_test =path2data(data,label,input_width,input_height,input_width,input_height)

y_pred = model.predict(X_test)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)

IoU(y_testi,y_predi)

visualize_prediction(X_test,y_predi,y_testi,classes)
print('finish visualization')
plot_history(hist_path)

