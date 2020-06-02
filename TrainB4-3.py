#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="test")
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
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
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

label_smooh_rate=0.2
Num_epochs=100
backbone="TrainB4"
batch_size = 128
chnnel_number = 3
root='/share_data/xc_1205_pure_expand'
img_height = img_width = 256
trainPath = os.path.join(root,'train')
valPath = os.path.join(root,'val')
testPath = os.path.join(root,'test')
#testPath = '/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1215/test_data/'
class_number = len(os.listdir(trainPath))
weight_path='/root/CEST_weights/{}'.format(backbone)
if not os.path.exists(weight_path):
    os.makedirs(weight_path) 
# determine the total number of image paths in training validation
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))      
train_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
    rescale=1. / 255,
    preprocessing_function=random_augimage()
    )
val_datagen = ImageDataGenerator(
        rotation_range=np.random.choice((0,45)),
        fill_mode='constant',
        cval=0,
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
test_generator = val_datagen.flow_from_directory(
    testPath,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=False,
    color_mode='rgb',
    class_mode='categorical')
print(test_generator.class_indices.keys())
class_label=['凹陷病变-阿弗他','凹陷病变-溃疡','凹陷病变-糜烂', '肠内容物-寄生虫','肠内容物-血液', '肠内容物-异物',
'隆起病变-结节', '隆起病变-静脉异常','隆起病变-息肉','隆起病变-肿块(肿瘤)','平坦病变-白斑', '平坦病变-点',
'平坦病变-红斑', '正常-无效-遮挡','正常-无杂质-翻折发白','正常-无杂质-黑色肠液','正常-无杂质-挤压发白','正常-无杂质-静脉', '正常-无杂质-黏膜红色',
'正常-无杂质-曝光发白','正常-无杂质-绒毛', '正常-杂质-白色固体','正常-杂质-白色漂浮','正常-杂质-黑色肠液', 
 '正常-杂质-挤压发白','正常-杂质-静脉','正常-杂质-弥散漂浮物','正常-杂质-曝光发白','正常-杂质-水泡','正常-杂质-LED灯'
]

val_metrics = Metrics(test_generator,validation_steps=(totalTest // batch_size) + 1,
                      save_path=weight_path,labels=class_label)

input_tensor = Input((img_height,img_width,chnnel_number))
#efnet=EfficientNetB4(weights='/root/fanweiya/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',include_top=False,input_tensor=input_tensor)
model=load_model('/root/CEST_weights/TrainB4/smr_0_17_valacc_0.9381_valloss_0.2300.h5')
model.summary()
plot_model(model,to_file=weight_path+'/model.png', show_shapes=True)
ppmodel = multi_gpu_model(model,gpus=2)
#loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
ppmodel.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-5),metrics=['accuracy'])
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,mode='auto')
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/smr_{}'.format(label_smooh_rate)+
                                     '_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
cbks = [early_stopping,checkpoint,val_metrics,CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=3e-4)]

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
history=ppmodel.fit_generator(train_generator,epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=32,
                                   verbose=1,callbacks=cbks)
'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.figure(figsize=(10,10))
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
