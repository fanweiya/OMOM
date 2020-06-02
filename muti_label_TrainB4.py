from efficientnet.keras import EfficientNetB0,EfficientNetB1,EfficientNetB3,EfficientNetB4
import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from imutils import paths
from keras import optimizers
import tensorflow as tf
from keras.utils import plot_model
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import losses
from losses import focal
import random
from keras import regularizers
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from keras.engine.topology import Layer
from keras.models import load_model
from keras import initializers
import itertools
import os
import cv2
from imutils import paths
import numpy as np
import segmentation_models as sm
from keras.layers import Concatenate
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams,font_manager
from consinlr import CosineAnnealingScheduler
from WRWD import *
from CyclicalLearningRate_callback import *
from SGDRScheduler import *
from sklearn.metrics import confusion_matrix
from keras.layers import Dense,Flatten,Input,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D,Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras import backend as K
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from randaugment import *
from metrics import *
from numpy.random import seed 
seed(1) 
from tensorflow import set_random_seed 
set_random_seed(2)
img_augment = Rand_Augment()
def random_augimage(p=0.5):
    def aug_iamge(input_img):
        aug_persion = np.random.rand()
        if aug_persion < p:
            return input_img
        else:
            input_img=img_to_array(img_augment(array_to_img(input_img)))
            return input_img
    return aug_iamge

train_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
    rescale=1. / 255,
    preprocessing_function=random_augimage()
    )
val_datagen = ImageDataGenerator(
        #rotation_range=np.random.choice((0,45)),
        #fill_mode='constant',
        #cval=0,
        rescale=1. / 255,
        horizontal_flip=np.random.choice((True,False)),
        vertical_flip=np.random.choice((True,False)))
gpu_num=1
import pandas as pd
import sqlite3
sql=pd.read_sql("SELECT * FROM data",con=sqlite3.connect('/root/supervising-ui-for-multi-label/workdir/db.sqlite'))
sql.dropna(subset=['label'], inplace=True)
df=sql[['url','label']]
from sklearn.utils import shuffle
df = shuffle(df)
label_smooh_rate=0
Num_epochs=100
backbone='TrainB4-mutilabel'
batch_size = 64*gpu_num
chnnel_number = 3
img_height = img_width = 256
weight_path='/root/CEST_weights/{}'.format(backbone)
if not os.path.exists(weight_path):
    os.makedirs(weight_path) 
class_label=['凹陷病变-阿弗他',
 '凹陷病变-溃疡',
 '凹陷病变-糜烂',
 '肠内容物-血液',
 '隆起病变-结节',
 '隆起病变-静脉异常',
 '隆起病变-息肉',
 '隆起病变-肿块(肿瘤)',
 '红色静脉',
 '灰蓝色静脉',
 '大杂质',
 '小杂质',
 '固体异物',
 '浑浊液',
 '大水泡',
 '小水泡',
 '密集水泡',
 '颜色异常',
 '缺水',
 '运动模糊',
 '镜头抵近']

class_number=len(class_label)
# *training* data
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# compute the training and testing split
train_num = int(df.shape[0] * TRAIN_SPLIT)
val_num = int(df.shape[0] * VAL_SPLIT)
traindf = df[:train_num]
valdf = df[train_num:train_num+val_num]
testdf=df[train_num+val_num:]

train_generator=train_datagen.flow_from_dataframe(
    dataframe=df[:train_num],
    x_col="url",
    y_col="label",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    classes=class_label,
    target_size=(img_height,img_width))

valid_generator=val_datagen.flow_from_dataframe(
    dataframe=df[train_num:],
    x_col="url",
    y_col="label",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    classes=class_label,
    target_size=(img_height,img_width))

test_generator=val_datagen.flow_from_dataframe(
    dataframe=df[train_num+val_num:],
    x_col="url",
    y_col="label",
    batch_size=batch_size,
    shuffle=False,
    class_mode="categorical",
    classes=class_label,
    target_size=(img_height,img_width))

input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB4(weights='/root/fanweiya/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',include_top=False,input_tensor=input_tensor)
flatten=Flatten()(efnet.get_layer('top_activation').output)
#dense1=Dense(512,activation='relu')(flatten)
#drop2=Dropout(0.5)(dense1,training=True)
prediction_output=Dense(class_number,activation='sigmoid')(flatten)
model=Model(efnet.inputs,prediction_output)
#model.summary()
if gpu_num>1: 
    Model = multi_gpu_model(model,gpus=gpu_num,cpu_merge=False)
else:
    Model= model

Model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-3,decay=1e-6),metrics=['accuracy'])

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss',patience=10,verbose=1,mode='auto')
cosin_lr=CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=3e-4)
WRWD_lr=WRWDScheduler(steps_per_epoch=np.ceil(train_generator.n / batch_size), lr=0.001, wd_norm=0.01)
SGDRS_lr=SGDRScheduler(min_lr=1e-5,max_lr=1e-2,steps_per_epoch=np.ceil(train_generator.n / batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
Cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=np.ceil(train_generator.n / batch_size), mode='triangular')
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/smr_{}'.format(label_smooh_rate)+'_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
cbks = [early_stopping,checkpoint,lr_reduce]

history=Model.fit_generator(train_generator,epochs=Num_epochs,
                                   steps_per_epoch=train_generator.n//train_generator.batch_size,
                                   validation_data=valid_generator,
                                   validation_steps=valid_generator.n//valid_generator.batch_size,
                                   use_multiprocessing = True,workers=32,class_weight='auto',
                                   verbose=1,callbacks=cbks)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.ylim(0, 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.subplot(122)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(weight_path+'/model-run-result.png')
#plt.show()
