#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="test")
from efficientnet.keras import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3
import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from imutils import paths
from keras import optimizers
import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
from keras import losses
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
from consinlr import CosineAnnealingScheduler
from metrics import *
from keras.constraints import MinMaxNorm
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
def smooth_positive_labels(inputs,smooth_factor):
    if 0 <= smooth_factor<= 1:
        K = inputs.shape[-1]
        smooth_lable=((1 - smooth_factor) * inputs) + (smooth_factor / K)
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))        
    return smooth_lable
label_smooh_rate=0
Num_epochs=50
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
class_label=['凹陷病变-阿弗他','凹陷病变-溃疡','凹陷病变-糜烂', '肠内容物-寄生虫','肠内容物-血液', '肠内容物-异物',
'隆起病变-结节', '隆起病变-静脉异常','隆起病变-息肉','隆起病变-肿块(肿瘤)','平坦病变-白斑', '平坦病变-点',
'平坦病变-红斑', '正常-无效-遮挡','正常-无杂质-翻折发白','正常-无杂质-黑色肠液','正常-无杂质-挤压发白','正常-无杂质-静脉', '正常-无杂质-黏膜红色',
'正常-无杂质-曝光发白','正常-无杂质-绒毛', '正常-杂质-白色固体','正常-杂质-白色漂浮','正常-杂质-黑色肠液', 
 '正常-杂质-挤压发白','正常-杂质-静脉','正常-杂质-弥散漂浮物','正常-杂质-曝光发白','正常-杂质-水泡','正常-杂质-LED灯'
]
val_metrics = Metrics(test_generator,validation_steps=(totalTest // batch_size) + 1,
                      save_path=weight_path,labels=class_label)

#base_model = load_model('/share_data/code/EfficientNetB3_weights_001_valacc_0.7409_valloss_0.8640.h5')
'''
input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB3(weights='imagenet',input_tensor=input_tensor)
Dense1=Dense(512,activation='relu')(efnet.get_layer('top_dropout').output)
drop1=Dropout(0.25)(Dense1)
Dense2=Dense(1024,activation='relu')(drop1)
drop2=Dropout(0.5)(Dense2)
prediction_output=Dense(class_number,activation='softmax')(drop2)
'''

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])

class WeightedSum(Layer):
    def __init__(self,**kwargs):
        super(WeightedSum, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='uniform',
                                      constraint = MinMaxNorm(0,1),
                                      trainable=True)
        K.set_value(self.kernel, np.array([0.5]))
        super(WeightedSum, self).build(input_shape)
    def call(self, two_model_outputs):
        return self.kernel * two_model_outputs[0] + (1 - self.kernel) * two_model_outputs[1]
    def compute_output_shape(self, input_shape):
        return input_shape[0]
unet_model = sm.Unet('efficientnetb0',
                     input_shape=(img_height, img_width, chnnel_number),
                     encoder_weights='imagenet',classes=1, activation='sigmoid',decoder_use_batchnorm=True)
for layer in unet_model.layers:
    layer.trainable = True
    layer.name = layer.name + str("_unet")
prediction1=Dense(class_number,activation='softmax',name='unet-pred')(Dropout(0.5)(Flatten()(unet_model.get_layer('top_activation_unet').output)))
cnn_dot_out = Lambda(batch_dot)([unet_model.input,unet_model.get_layer(index=-1).output])
#concate=Concatenate()([base_model.input,base_model.get_layer(index=-1).output])
effb3 = EfficientNetB3(input_shape=(img_height,img_width,int(cnn_dot_out.shape[-1])),weights=None,include_top=False)(cnn_dot_out)
prediction2=Dense(class_number,activation='softmax',name='main-pred')(Dropout(0.5)(Flatten()(effb3)))


final_pred=WeightedSum()([prediction1,prediction2])
with tf.device('/cpu:0'):
    model=Model(inputs=unet_model.input,outputs=final_pred)
model.summary()
#model.load_weights('/root/CEST_weights/test_paper-3/smr_0_09_valacc_0.9066_valloss_0.3800.h5')
ppmodel = multi_gpu_model(model,gpus=2)
ppmodel.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy'])
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
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/weight2sum_smr_{}'.format(label_smooh_rate)+
                                     '_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
cbks = [early_stopping,checkpoint,CosineAnnealingScheduler(T_max=10, eta_max=1e-3, eta_min=3e-4),val_metrics]
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
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.figure(figsize=(10,20))
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
