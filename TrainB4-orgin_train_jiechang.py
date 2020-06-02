#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="test")
#coding=utf8
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
gpu_num=1
label_smooh_rate=0
Num_epochs=100
backbone='TrainB4-newdata'
batch_size = 64*gpu_num
chnnel_number = 3
root='/share_data/XC_V1_expand_jiechang'
img_height = img_width = 256
trainPath = os.path.join(root,'train')
valPath = os.path.join(root,'val')
class_number = len(os.listdir(trainPath))
weight_path='/root/CEST_weights/{}'.format(backbone)
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

class_label=['阿弗他-无杂质', '阿弗他-杂质','闭合结肠','多节结肠',
           '发红-像阿弗他','光滑结肠','黑色肠液','环状结肠','黄色黏膜','回盲瓣+结肠','角结肠','结肠','结肠黏膜',
           '结节-非局灶-无杂质', 
           '结节-非局灶-杂质', '结节-局灶-无杂质',
           '结节-局灶-杂质', '结节-无效', '结节息肉待审',
           '静脉异常隆起-有杂质', '溃疡-非局灶-有杂质', 
           '溃疡-局灶-无杂质', '溃疡-局灶-杂质', '蓝色静脉-无杂质', 
           '蓝色静脉-有杂质','阑尾口','糜烂-非局灶-伴随发红-无杂质',
           '糜烂-非局灶-伴随发红-有杂质', '糜烂-非局灶-无杂质',
           '糜烂-非局灶-杂质', '糜烂-局灶-暗黄色凹陷-无杂质',
           '糜烂-局灶-暗黄色凹陷-杂质', '糜烂-局灶-伴随发红-无杂质', 
           '糜烂-局灶-伴随发红-有杂质', '糜烂-局灶-无杂质', '糜烂-局灶-杂质',
           '糜烂-类圆形-无杂质', '糜烂-类圆形-有杂质', '其他-盒子',
           '其他-脸', '其他-手', '其他-衣服','三角超近结肠','三角结肠','无效1', '无效10',
           '无效11', '无效12', '无效13', '无效14', '无效15',
           '无效17', '无效18', '无效19', '无效2', '无效3', '无效4', '无效5',
           '无效6', '无效7', '无效8', '无效9','无效_多残渣','无效_多水泡','无效_过浑浊','无效_过曝光','无效_一半结肠一半粪便','息肉-非局灶', '息肉-局灶',
           '血块-黑色', '血块-红色', '血液-非局灶-暗红色', '血液-非局灶-黑绿色',
           '血液-非局灶-红色', '血液-非局灶-红色血渣', '血液-局灶-红色',
           '血液-局灶血丝', '血液-无效','褶皱结肠','正常-翻折发白', '正常-无杂质-红色静脉',
           '正常-无杂质-黏膜红色', '正常-无杂质-绒毛',
           '正常-无杂质-绒毛颗粒', '正常-杂质-红色静脉', 
           '正常-杂质-黄白漂浮物', '正常-杂质-灰蓝色静脉',
           '正常-杂质-弥散漂浮物', '正常-杂质-水泡', 
           '肿块(肿瘤)-伴随发红-有杂质', '肿块(肿瘤)-局灶-无杂质',
           '肿块(肿瘤)-局灶-有杂质', 
           '肿块(肿瘤)-疑似-无杂质', '肿块(肿瘤)-疑似-杂质']

val_metrics = Metrics(validation_generator,validation_steps=(totalVal // batch_size) + 1,
                      save_path=weight_path,labels=class_label)
input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB4(weights='/root/fanweiya/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',include_top=False,input_tensor=input_tensor)
avgpool=GlobalAveragePooling2D()(efnet.output)
drop1=Dropout(0.5)(avgpool)
prediction_output=Dense(class_number,activation='softmax')(drop1)
model=Model(efnet.inputs,prediction_output)
model.summary()
plot_model(model,to_file=weight_path+'/model1.png', show_shapes=True)
if gpu_num>1: 
    Model = multi_gpu_model(model,gpus=gpu_num,cpu_merge=False)
else:
    Model= model
#loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
Model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3,momentum = 0.9, nesterov = True),metrics=['accuracy'])
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',factor = 0.5, min_lr = 1e-6, verbose = 1)
cosin_lr=CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=1e-6)
WRWD_lr=WRWDScheduler(steps_per_epoch=np.ceil(totalTrain / batch_size), lr=0.001, wd_norm=0.01)
SGDRS_lr=SGDRScheduler(min_lr=1e-5,max_lr=1e-2,steps_per_epoch=np.ceil(totalTrain / batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
Cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=np.ceil(totalTrain / batch_size), mode='triangular')
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/jc_smr_{}'.format(label_smooh_rate)+'_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
cbks = [early_stopping,checkpoint,lr_reduce,val_metrics,cosin_lr]
'''
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
history=ppmodel.fit_generator(gen_smooth_labels(train_generator),epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=gen_smooth_labels(validation_generator),
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=32,
                                   verbose=1,callbacks=cbks)
'''
history=Model.fit_generator(train_generator,epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=32,
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
plt.savefig(weight_path+'/model-run-result1.png')
