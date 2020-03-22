'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 14:28:17
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-14 15:34:03
'''
import cv2
import numpy as np
import keras
import math
#import tensorflow as tf
#from tensorflow.python.keras.utils.data_utils import Sequence
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, datas, labels,nClasses=21,batch_size=32, shuffle=True,input_shape=[224,224],output_shape=[224,224]):
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.output_width = output_shape[0]
        self.output_height = output_shape[1]
        self.nClasses = nClasses

    def __len__(self):
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, idx):

        batch_indexs = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_labels = [self.labels[k] for k in batch_indexs]
        #batch_datas = self.datas[idx * self.batch_size:(idx + 1)*self.batch_size]
        #batch_labels = self.labels[idx * self.batch_size:(idx + 1)*self.batch_size]
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
            images.append(getImageArr(im, self.input_width , self.input_height))
            labels.append(getSegmentationArr(seg, self.nClasses , self.output_width , self.output_height))
       
        return np.array(images), np.array(labels)



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

def read_data_path(root='/home/throne/data/VOCdevkit/VOC2012',extension='/list/trainval_aug.txt', train=True):
    txt_fname = root + extension
    with open(txt_fname, 'r') as f:
        images = f.read().splitlines()
    print(len(images))
    data = [root+i.split()[0] for i in images]
    label = [root+i.split()[1] for i in images]
    return data, label

def divider(X,y,train_rate=0.85):
    index_train = np.random.choice(len(X),int(len(X)*train_rate),replace=False)
    index_test  = list(set(range(len(X))) - set(index_train))
    X_train, y_train = np.array(X)[index_train],np.array(y)[index_train]
    X_test, y_test = np.array(X)[index_test],np.array(y)[index_test]
    return X_train,y_train,X_test,y_test

def path2data(X,y,input_width=224,input_height=224,output_width=224,output_height=224):
    X_data=[]
    y_label=[]
    for im,seg in zip(X,y) :
        X_data.append(getImageArr(im , input_width , input_height))
        y_label.append(getSegmentationArr(seg , 21 , output_width , output_height))
    return np.array(X_data),np.array(y_label)
