
from keras.utils import Sequence
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import random 
import os
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.utils import Sequence

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import random 
import os

class PatchAugmentation(Sequence):
    def __init__(self,generator, probability=1.0, min_patch_dimension=0.1, 
                 max_patch_dimension=0.9, image_area=256*256):
        self.batch_size = generator.batch_size
        #print(self.batch_size)
        self.x_train = generator.next()[0]
        self.y_train = generator.next()[1]
        self.probability = probability
        self.image_area = generator.target_size[0]**2
        #print(self.image_area)
        self.min_patch_dimension = min_patch_dimension
        self.max_patch_dimension = max_patch_dimension
        self.dim = int(round(math.sqrt(self.image_area)))

    def __len__(self):
        return int(np.ceil(len(self.x_train) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        batch_x = np.copy(self.x_train[idx * self.batch_size:(idx+1) * self.batch_size])
        batch_y = np.copy(self.y_train[idx * self.batch_size:(idx+1) * self.batch_size])
        
        for i in range(len(batch_x)):
            
            if np.random.uniform(0, 1) <= self.probability:
                
                # Get the minimum width and maximum width of a patch
                min_width = round(self.min_patch_dimension * self.dim)
                max_width = round(self.max_patch_dimension * self.dim)
                
                # Get the minimum width and maximum width of a patch
                min_height = round(self.min_patch_dimension * self.dim)
                max_height = round(self.max_patch_dimension * self.dim)
                
                # Get the random patch dimensions, between min/max height and width
                horizontal_dim = np.random.randint(min_width, max_width+1)
                vertical_dim = np.random.randint(min_height, max_height+1)
                
                # Get a random location from where to take the patch
                # First, get the bounds where the patch can be taken 
                # (must be possible to extract this patch)
                x1 = np.random.randint(0, self.dim - horizontal_dim)
                y1 = np.random.randint(0, self.dim - vertical_dim)
                x2 = x1 + horizontal_dim
                y2 = y1 + vertical_dim
                 
                # Generate placement co-ordinates
                x1p = np.random.randint(0, self.dim - horizontal_dim)
                y1p = np.random.randint(0, self.dim - vertical_dim) 
                x2p = x1p + horizontal_dim
                y2p = y1p + vertical_dim
                
                # Get a random sample from the entire training set 
                # This means we could be taking from the same class!
                r_i = np.random.randint(0, len(self.x_train))
                
                batch_x[i][x1p:x2p, y1p:y2p, :] = self.x_train[r_i][x1:x2, y1:y2, :]
                                
                lambda_value = (horizontal_dim * vertical_dim) / (self.dim * self.dim)
                batch_y[i] = (1- lambda_value) * batch_y[i] + lambda_value * self.y_train[r_i]
            
        return batch_x, batch_y
'''
patch_swap_generator = PatchAugmentation(test_generator)
'''


'''
class PatchAugmentation(Sequence):
    def __init__(self, batch_size, x, y, probability=1.0, image_area=1024, patch_area=0.25):
        self.batch_size = batch_size
        self.x_train = x
        self.y_train = y
        self.probability = probability
        self.image_area = image_area
        self.patch_area = patch_area
        
        # Calculate various dimensions for the patch placement.
        self.dim = int(round(math.sqrt(self.image_area)))
        self.crop_area_in_pixels = self.image_area * self.patch_area
        self.crop_dim = int(round(math.sqrt(self.crop_area_in_pixels)))
        self.max_horizontal_shift = (math.sqrt(self.image_area)) - self.crop_dim
        self.max_vertical_shift = (math.sqrt(self.image_area)) - self.crop_dim
        
    def __len__(self):
        return int(np.ceil(len(self.x_train) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        batch_x = np.copy(self.x_train[idx * self.batch_size:(idx+1) * self.batch_size])
        batch_y = np.copy(self.y_train[idx * self.batch_size:(idx+1) * self.batch_size])
        
        for i in range(len(batch_x)):
            
            if np.random.uniform(0, 1) <= self.probability:

                r_i = np.random.randint(0, len(self.x_train))
                
                x1 = np.random.randint(0, self.dim - self.crop_dim)
                x2 = x1 + self.crop_dim
                y1 = np.random.randint(0, self.dim - self.crop_dim)
                y2 = y1 + self.crop_dim

                batch_x[i][x1:x2, y1:y2, :] = self.x_train[r_i][x1:x2, y1:y2, :]
                                
                lambda_value = self.patch_area
                batch_y[i] = (1- lambda_value) * batch_y[i] + lambda_value * self.y_train[r_i]
            
        return batch_x, batch_y

patch_swap_generator = PatchAugmentation(batch_size=128, 
                                         x=x_train, 
                                         y = y_train, 
                                         probability=0.5, 
                                         patch_area=0.25)
'''
