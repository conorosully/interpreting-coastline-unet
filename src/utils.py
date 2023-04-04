
#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from osgeo import gdal

import os
import glob
import random
import torch

class model_eval:

    def __init__(self, model, images,targets, preds):
        self.model = model
        self.images = images
        self.targets = targets
        self.preds = preds
        self.fig_path = '/Users/conorosullivan/Google Drive/My Drive/UCD/research/Interpreting UNet/Figures/{}.png'
        self.device = torch.device('mps') 

    def calc_accuracy(self,pred, target):
        """Returns accuracy of model prediction"""
        
        correct = np.sum(target == pred)
        h,w = target.shape

        accuracy = correct/(h*w)
            
        return correct, accuracy
    
    def get_rgb(self,i):
        """Return normalized RGB channels from sentinal image"""
        img = self.images[i].cpu().detach().numpy()

        rgb = img[[3,2,1]]
        rgb = rgb.transpose(1,2,0) 
        rgb = np.clip(rgb, 0, 0.3)/0.3

        return rgb

    def plot_pred(self,i,save=None):
        """Diplays RGB image, label and prediction with accuracy"""
        rgb = self.get_rgb(i)
        target = self.targets[i]
        pred = self.preds[i]

        fig,axs = plt.subplots(1,3,figsize=(15,15))
        fig.patch.set_facecolor('xkcd:white')

        axs[0].imshow(rgb)
        axs[0].set_title("RGB Image",size=15)

        axs[1].imshow(target,cmap="gray")
        axs[1].set_title("Target",size=15)

        axs[2].imshow(pred,cmap="gray")
        correct, accuracy = self.calc_accuracy(pred, target)
        accuracy = round(accuracy*100, 2)
        axs[2].set_title("Accuracy: {}%".format(accuracy))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        if save:
            plt.savefig(self.fig_path.format(save),bbox_inches='tight',dpi=300)

    
    def confusion_metrics(self,pred,target):
        """Returns confusion matrix metrics"""
        
        TP = np.sum((pred == 1) & (target == 1))
        TN = np.sum((pred == 0) & (target == 0))
        FP = np.sum((pred == 1) & (target == 0))
        FN = np.sum((pred == 0) & (target == 1))
        
        return TP,TN,FP,FN
    
    def eval_metrics(self):
        """Evaluate model performance on test set"""

        TP,TN,FP,FN =0,0,0,0
        r_accuracy = []
        r_balanced_accuracy = []
        r_precision = []
        r_recall = []
        r_f1 = []

        # Calcualte accuracy
        for i in range(len(self.targets)):
            target = self.targets[i]
            pred = self.preds[i]

            TP_,TN_,FP_,FN_ = self.confusion_metrics(pred,target)
            TP += TP_
            TN += TN_
            FP += FP_
            FN += FN_

            r_accuracy.append((TP_+TN_)/(TP_+TN_+FP_+FN_))
            r_balanced_accuracy.append(0.5*(TP_/(TP_+FN_) + TN_/(TN_+FP_)))
            r_precision.append(TP_/(TP_+FP_))
            r_recall.append(TP_/(TP_+FN_))
            r_f1.append(2*(r_precision[-1]*r_recall[-1])/(r_precision[-1]+r_recall[-1]))

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        balanced_accuracy = 0.5*(TP/(TP+FN) + TN/(TN+FP))
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)

        return {'accuracy':r_accuracy,
                'balanced_accuracy':r_balanced_accuracy,
                'precision':r_precision,
                'recall':r_recall,
                'f1':r_f1},{'accuracy':accuracy,
                'balanced_accuracy':balanced_accuracy,
                'precision':precision,
                'recall':recall,
                'f1':f1}
    
    def plot_importance(self,importance,channels_,label=None,save=None):
        
        fig, ax  = plt.subplots(figsize=(10,4))
        fig.set_facecolor('white')


        plt.bar(height=importance,x=np.arange(0,len(channels_)),)

        plt.ylabel(label,size=20)

        ax.set_xticks(np.arange(0,len(channels_)),channels_,rotation=90, size=15)
        ax.set_yscale('log')

        plt.yticks(size=15)

        if save:
            plt.savefig(self.fig_path.format(save),bbox_inches='tight',facecolor='white', dpi=300)

    def get_weights(self):
        """Returns average weight of input layer"""

        weight = self.model.e1.conv.conv1.weight
        weight = weight.detach().cpu().numpy()
        weight = np.abs(weight)
        avg_weight = weight.transpose(1,0,2,3).mean(axis=(1,2,3))
    
        return avg_weight
    
    def shuffle_band(self,band):
        """Shuffle a single spectral band of an image"""

        perm_image = band.ravel()
        random.shuffle(perm_image)
        perm_image.resize(256,256)

        return perm_image

    def permutate_bands(self,img,bands):
        """Permuate bands in list of bands"""

        img = img.copy()

        band_dict = {'Coastal Aerosol':0,'Blue':1,'Green':2,
                    'Red':3,'Red Edge 1':4,'Red Edge 2':5,
                    'Red Edge 3':6,'NIR':7,'Red Edge 4':8,
                    'Water Vapour':9,'SWIR 1':10,'SWIR 2':11}

        bands_i = [band_dict[b] for b in bands]

        for i in bands_i:
            img[i,:,:] = self.shuffle_band(img[i,:,:])

        return img
    
    def get_pred(self,output):
        """Get prediction from model output"""

        pred = output.cpu().detach().numpy()[1]
        pred = np.round(pred)

        return pred

    def get_perm_accuracy(self,bands):
        """Get accuracy of model on permuated spectral bands"""

        orignal_images = self.images.cpu().detach().numpy()

        perm_images = orignal_images.copy()

        for i in range(len(orignal_images)):
            perm_images[i] = self.permutate_bands(perm_images[i],bands)
                

        perm_images = np.array(perm_images)
        perm_images = torch.from_numpy(perm_images)
        perm_images = perm_images.to(self.device)

        # get model predcitions
        output = self.model(perm_images)
        perm_preds = np.array([self.get_pred(x) for x in output])
        
        # Calc accuracy
        r_targets = np.array(self.targets)
        x,y,z = np.shape(r_targets)
        accuracy = np.sum(r_targets == perm_preds)/(x*y*z)

        return accuracy
        

    
                         

