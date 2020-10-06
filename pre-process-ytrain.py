def Y_dataframe(Y_values):

 '''
Input : It is a pandas dataframe that countains the lables and bounding boxes of each image in the file.

Output: Y and Y_keys 
        Y_ keys is a  dataframe that has all the labels.
        Y is a  dataframe that holds all the one hot codes and its respective labels. 
 '''
 
 Y_values[0] = Y_values[0].str[:9]
 Y_values.drop(Y_values.index[:1], inplace=True)

 # Key for the dataset 
 image_types = Y_values[0].unique()
 Y_train = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y_train['Image_Types_labels'] = labelencoder.fit_transform(Y_train['Image_Types'])
 Y_keys = Y_train
 
 image_types = np.array(Y_values[0])
 Y = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y['Image_Types_labels'] = labelencoder.fit_transform(Y['Image_Types'])
 
    
 enc = OneHotEncoder(handle_unknown='ignore')
 enc_df = pd.DataFrame(enc.fit_transform(Y[['Image_Types_labels']]).toarray())
 Y = Y.join(enc_df)
 
 return Y,Y_keys
 
 

def dow_val():
 Y_values = pd.read_csv('../input/image-detect/val/val_annotations.txt',delimiter="\t",header = None)
# Y_values = Y_values.sort_values([0])

 # Key for the dataset 
 image_types = Y_values[1].unique()
 Y_train = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y_train['Image_Types_labels'] = labelencoder.fit_transform(Y_train['Image_Types'])
 Y_keys = Y_train

 image_types = np.array(Y_values[1])

 Y = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
    
 Y['Image_Types_labels'] = labelencoder.fit_transform(Y['Image_Types']) 
 enc = OneHotEncoder(handle_unknown='ignore')
 enc_df = pd.DataFrame(enc.fit_transform(Y[['Image_Types_labels']]).toarray())
 Y = Y.join(enc_df)
 
 return Y,Y_keys
 
 
 #Separating hot encode from y values
 
 def hot_encode(X_train):
 arra = X_train[X_train.columns[2:]]
 arra = np.array(arra)
 return arra
 
#Setting up data from functions

# First downloading data from drive 
y_train,X_train = download('../input/image-detect/train')
y_val,X_val = download('../input/image-detect/val')

# Assembling the Y values in the right order
y_train,y_train_keys = Y_dataframe(y_train)
y_val,y_val_keys =  dow_val()

y_val,X_val = download('../input/image-detect/val')

for (root,dirs,files) in os.walk('../input/image-detect/val/images'):
    print(files)
    
Y_train = hot_encode(y_train)
Y_val = hot_encode(y_val)
X_train = X_train/255
X_val = X_val/255

save('X_val_unsortted.npy',X_val)
save('Y_val_unsortted',Y_val)

print(X_train.dtype)

tf.dtypes.cast(X_train, tf.float16)
print(X_train.dtype)


#Downloading pre processed data

import numpy as np 
import tensorflow as tf
from numpy import load
import pandas as pd
import gc

#Downloading X_train,X_val,Y_train,Y_val
X_train = load('../input/pre-trained-data/X_train.npy')
X_val = load('../input/last-data/X_train(3).npy')/255
Y_train = load('../input/pre-trained-data/Y_train.npy')
Y_val = load('../input/validation/Y_val_unsortted.npy')

#Dowloading Key values dataframe 
Y_train_keys = pd.read_csv('../input/pre-trained-data/y_train_keys.txt',delimiter="\t")
Y_val_keys =pd.read_csv('../input/pre-trained-data/y_val_keys.txt',delimiter="\t")
