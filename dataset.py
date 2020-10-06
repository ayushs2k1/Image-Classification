#importing the libraries

import numpy as np 
import tensorflow as tf
import cv2
import os 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import save


#Downloading the data

  '''
  Input: The path for the dataset
  
  Output: y_train ; It is a pandas dataframe consisting of the y labels.
          x_train ; It is a tensor with image data.
  '''
  data = pd.DataFrame({0:['n99999999_001.JPEG'], 
                       1:[5],
                       2:[0],
                       3:[0]}) 
  image = None
  for (root,dirs,files) in os.walk(path): 
    if(root != path):
      if(dirs == []):
        for file in files:
          print(file)
          if(image != None):
            a = root+'/' + file
            image_raw = tf.io.read_file(a)
            new = tf.image.decode_image(image_raw)
            if(tf.shape(new)[2] == 1):
              new = tf.image.grayscale_to_rgb(new, name=None)
            new = tf.reshape(new,[1,64,64,3])
            image = tf.concat([image,new],0)
    
          
          else:
            a = root+'/' + file
            image_raw = tf.io.read_file(a)
            image = tf.image.decode_image(image_raw)
            if(tf.shape(image)[2] == 1):
              image = tf.image.grayscale_to_rgb(image, name=None)
            image = tf.reshape(image,[1,64,64,3])
     
     
      else:
        for dir in files:
          a = root + '/' + dir 
          data = data.append(pd.read_csv(a,delimiter="\t",header = None),ignore_index = True)
          
  y_train = data 
  x_train = image
  return y_train,x_train
