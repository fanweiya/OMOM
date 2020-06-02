from efficientnet.keras import EfficientNetB0,EfficientNetB1,EfficientNetB3,EfficientNetB7
import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from imutils import paths
from keras import optimizers
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import plot_model
config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import losses
from losses import categorical_focal_loss
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
from keras.layers import Concatenate
from consinlr import CosineAnnealingScheduler
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams,font_manager
from keras_radam import RAdam
from keras_lookahead import Lookahead
from sklearn.metrics import confusion_matrix
from keras.layers import Dense,Flatten,Input,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D,Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,LearningRateScheduler
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras import backend as K
from keras.utils import plot_model
from keras.utils import multi_gpu_model
import time
from keras.callbacks import Callback
from randaugment import *
from metrics import *
from numpy.random import seed 
seed(1) 
from tensorflow import set_random_seed 
set_random_seed(2)
def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d-%H:%M:%S', local_time)
    return str_time

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
gpu_num=3
label_smooh_rate=0
Num_epochs=100
backbone='TrainB7-newdata'
batch_size = 16*gpu_num
chnnel_number = 3

root='/share_data/PRDATA/20200512_训练集_expand/20200512_训练集_expand'
img_height = img_width = 256
trainPath = os.path.join(root,'train')
valPath = os.path.join(root,'val')
class_number = len(os.listdir(trainPath))
weight_path='/share_data/PRDATA/fanweiya/CEST_weights/{}'.format(backbone)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
weight_path=weight_path+'/model_'+get_current_time()
if not os.path.exists(weight_path):
    os.makedirs(weight_path)  
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))     
train_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
    rescale=1. / 255,
    width_shift_range=np.random.choice(np.linspace(0, 0.1,5)),
    height_shift_range=np.random.choice(np.linspace(0, 0.1,5)),
    preprocessing_function=random_augimage()
    )
val_datagen = ImageDataGenerator(
        #rotation_range=np.random.choice((0,45)),
        #fill_mode='constant',
        #cval=0,
        rescale=1. / 255,
        horizontal_flip=np.random.choice((True,False)),
        vertical_flip=np.random.choice((True,False)))

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
    color_mode='rgb',
    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
    valPath,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical')

class_label=list(train_generator.class_indices.keys())
def save_labelfile(labelpath,label):
    labeljs = os.path.join(labelpath,'label.js')
    if not os.path.exists(labeljs):
        try:
            os.system(r"touch {}".format(labeljs))
            f=open(labeljs,'w')
            f.write(str(label))
            f.close()
        except:
            pass

save_labelfile(weight_path,class_label)
val_metrics = Metrics(validation_generator,validation_steps=(totalVal // batch_size) + 1,
                      save_path=weight_path,labels=class_label)
'''
input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB7(weights='/share_data/PRDATA/fanweiya/beifen/fanweiya/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',include_top=False,input_tensor=input_tensor)
avgpool=GlobalAveragePooling2D()(efnet.output)
drop1=Dropout(0.5)(avgpool)
prediction_output=Dense(class_number,activation='softmax')(drop1)
model=Model(efnet.inputs,prediction_output)
#model.summary()
'''
model=load_model('/share_data/PRDATA/fanweiya/CEST_weights/TrainB7-newdata/model_2020-05-29-12:46:14/smr_0_03_valacc_0.8170_valloss_0.4562.h5')
#base_model.layers.pop()
#prediction_output=Dense(class_number,activation='softmax')(base_model.layers[-1].output)
#model=Model(base_model.inputs,prediction_output)

model.summary()

#plot_model(model,to_file=weight_path+'/model.png', show_shapes=True)

if gpu_num>1: 
    Model = multi_gpu_model(model,gpus=gpu_num,cpu_merge=False)
else:
    Model= model

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
early_stopping = EarlyStopping(monitor='val_loss',patience=8,verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,verbose=1, mode='auto', epsilon=0.0001)
#cosin_lr=CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=1e-6)
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/smr_{}'.format(label_smooh_rate)+'_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',save_best_only=True,verbose=1)
tensorboard_log=TensorBoard(log_dir=weight_path+'/model_log')

cbks = [early_stopping,checkpoint,lr_reduce,val_metrics,tensorboard_log]

def smooth_positive_labels(inputs,smooth_factor):
    if 0 <= smooth_factor<= 1:
        K = inputs.shape[-1]
        smooth_lable=((1 - smooth_factor) * inputs) + (smooth_factor / K)
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))        
    return smooth_lable
def gen_smooth_labels(datagen):
    while True:
        iamge,label = datagen.next()
        yield iamge,smooth_positive_labels(label,label_smooh_rate)

for layer in Model.layers[:-3]:  
    layer.trainable = False
RAdam_optimizers = RAdam(learning_rate=1e-3)
Model.compile(loss='categorical_crossentropy',optimizer=RAdam_optimizers,metrics=['acc'])
history=Model.fit_generator(train_generator,epochs=3,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=24,class_weight='auto',
                                   verbose=1,callbacks=cbks)

model=load_model(os.path.join(weight_path,os.listdir(weight_path)[-1]))

#optimizers = optimizers.SGD(lr=lr_schedule(0)*gpu_num,momentum = 0.9, nesterov = True)
for layer in Model.layers:  
    layer.trainable = True
Model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=1e-3),metrics=['acc'])
'''
history=Model.fit_generator(gen_smooth_labels(train_generator),epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=gen_smooth_labels(validation_generator),
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=256,
                                   verbose=1,callbacks=cbks)

'''
history=Model.fit_generator(train_generator,epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,use_multiprocessing = True,workers=24,
                                   validation_steps=totalVal // batch_size,
                                   verbose=1,callbacks=cbks)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(val_acc))
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
plt.savefig(weight_path+'/model-run-result1.png')
