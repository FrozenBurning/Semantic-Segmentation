'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 14:22:44
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-14 16:05:35
'''
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
sns.set_style("whitegrid", {'axes.grid' : False})


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

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))

def visualize_prediction(X_test,y_predi,y_testi,nClasses,num=12):
    for i in range(num):
        img_is  = (X_test[i] + 1)*(255.0/2)
        seg = y_predi[i]
        segtest = y_testi[i]

        fig = plt.figure(figsize=(10,30))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(img_is/255.0)
        ax.set_title("original")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(give_color_to_seg_img(seg,nClasses))
        ax.set_title("predicted class")

        ax = fig.add_subplot(1,3,3)
        ax.imshow(give_color_to_seg_img(segtest,nClasses))
        ax.set_title("true class")
        plt.savefig(str(i)+'.png')
    return 

def plot_history(historian):
    plt.figure()
    with open(historian,'rb') as file_pi:
        hist=pickle.load(file_pi)

    for key in ['loss', 'val_loss']:
        plt.plot(hist[key],label=key)
    plt.legend()
    plt.savefig('loss.png')

