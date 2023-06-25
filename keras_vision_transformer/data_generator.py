import os
import tensorflow as tf
from glob import glob
import numpy as np
import cv2

class DGS2S(tf.keras.utils.Sequence):
    def __init__(self, filepath:str, shuffle:bool=True, img_size:list=[128,128], batch_size:int=1, in_channels:int=1, out_category:int=1) -> None:
        self.shuffle = shuffle
        self.in_names = np.array(sorted(glob(os.path.join(filepath, 'images','*.png'))))
        self.batch_size=  batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_category = out_category
        self.N = len(self.in_names)
        assert self.N > 0
        pass
        
    def augment(self, in_img, out_img):
        if self.shuffle:
            x = np.random.randint(0,in_img.shape[1]-self.img_size[1])
            y = np.random.randint(0,in_img.shape[0]-self.img_size[0])
        else:
            x,y = 0,0

        X = in_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        Y = out_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        return X,Y

    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.in_channels), np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.out_category), np.float32)
        for ib in range(self.batch_size):
            i = (index*self.batch_size + ib)%(self.N)
            img = cv2.imread(self.in_names[i], 0).astype(np.float32)/255
            H, W = img.shape
            in_img, out_img = img[:,:(W//2)], img[:,(W//2):]
            in_img, out_img = self.augment(in_img, out_img)

            X[ib,:,:,0] = in_img
            Y[ib,:,:,0] = out_img
        return X,Y
    
    def __len__(self):
        return self.N*100


class DGStyle(tf.keras.utils.Sequence):
    def __init__(self, filepath:str, shuffle:bool=True, img_size:list=[128,128], batch_size:int=1, in_channels:int=2, out_category:int=1) -> None:
        self.shuffle = shuffle
        self.in_names = np.array(sorted(glob(os.path.join(filepath, 'images','*.png'))))
        self.batch_size=  batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_category = out_category
        self.N = len(self.in_names)
        assert self.N > 0
        pass
        
    def augment(self, in_img, out_img):
        if self.shuffle:
            x = np.random.randint(0,in_img.shape[1]-self.img_size[1])
            y = np.random.randint(0,in_img.shape[0]-self.img_size[0])
        else:
            x,y = 0,0

        X = in_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        Y = out_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        return X,Y

    def augment_intensity(self, in_img, out_img, scale_min:float=0.5, scale_max:float=1.5, offset_min:float=-0.1, offset_max:float=0.1):
        if self.shuffle:
            scale = scale_min + np.random.rand()*(scale_max - scale_min)
            offset = offset_min + np.random.rand()*(offset_max - offset_min)
        else:
            scale,offset = 1.0,0.0



        X = in_img
        Y = out_img*scale + offset
        return X,Y
    
    def augment_blur(self, in_img, out_img, sigma_max:float=2.0):
        if self.shuffle:
            sigma = sigma_max * np.random.rand()
            
            
        else:
            sigma = 0.0

        if sigma>0.1:
            Y = cv2.GaussianBlur(out_img, (5,5), sigma)
        else:
            Y = out_img

        X = in_img
        return X,Y
    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.in_channels), np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.out_category), np.float32)
        for ib in range(self.batch_size):
            i = (index*self.batch_size + ib)%(self.N)
            img = cv2.imread(self.in_names[i], 0).astype(np.float32)/255
            H, W = img.shape
            in_img, out_img = img[:,:(W//2)], img[:,(W//2):]
            in_img, out_img = self.augment(in_img, out_img)
            in_img, out_img = self.augment_intensity(in_img, out_img)
            in_img, out_img = self.augment_blur(in_img, out_img)

            in_img = np.concatenate([in_img[:,:,np.newaxis], out_img[:,:,np.newaxis]], axis=-1)
            out_img = out_img[:,:,np.newaxis]
            

            X[ib,:,:,:] = in_img
            Y[ib,:,:,:] = out_img
        return X,Y
    
    def __len__(self):
        return self.N*100
    

class DG(tf.keras.utils.Sequence):
    def __init__(self, filepath:str, shuffle:bool=True, img_size:list=[128,128], batch_size:int=1, in_channels:int=3, out_category:int=2) -> None:
        self.shuffle = shuffle
        self.in_names = np.array(sorted(glob(os.path.join(filepath, 'images','*.png'))))
        self.batch_size=  batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_category = out_category
        self.N = len(self.in_names)
        assert self.N > 0
        pass
        
    def augment(self, in_img, out_img):
        if self.shuffle:
            x = np.random.randint(0,in_img.shape[1]-self.img_size[1])
            y = np.random.randint(0,in_img.shape[0]-self.img_size[0])
        else:
            x,y = 0,0

        X = in_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        Y = out_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        return X,Y

    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.in_channels), np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.out_category), np.float32)
        for ib in range(self.batch_size):
            i = (index*self.batch_size + ib)%(self.N)
            in_img = cv2.imread(self.in_names[i]).astype(np.float32)/255
            out_img = cv2.imread(self.out_names[i],0).astype(np.float32)/255
            in_img, out_img = self.augment(in_img, out_img)

            out_img = tf.keras.utils.to_categorical(np.round(out_img).astype(np.int32),self.out_category)
            X[ib,...] = in_img
            Y[ib,...] = out_img
        return X,Y
    
    def __len__(self):
        return self.N*100
