from randaugment import Rand_Augment
#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="test")
from efficientnet.keras import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from imutils import paths
from keras import optimizers
import tensorflow as tf
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
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams,font_manager
from sklearn.metrics import confusion_matrix
from keras.layers import Dense,Flatten,Input,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D
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
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import plot_model
import numpy as np
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from randaugment import Rand_Augment
from consinlr import CosineAnnealingScheduler
from keras.callbacks import Callback
from itertools import tee
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,classification_report
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
plt.figure(figsize=(20,20))
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname='/root/simhei.ttf',size=14)
sns.set(font=myfont.get_name())
plt.tight_layout()
def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]

class Metrics(Callback):
    def __init__(self, validation_generator, validation_steps,save_path,labels):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.save_path=save_path
        self.labels=labels
    def on_train_begin(self, logs={}):
        self.val_f1_scores = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        # duplicate generator to make sure y_true and y_pred are calculated from the same observations
        gen_1, gen_2 = tee(self.validation_generator)
        y_true = (np.vstack(next(gen_1)[1] for _ in range(self.validation_steps))).argmax(axis=1)
        tta_steps = 3
        preds_tta = []
        for i in tqdm(range(tta_steps)):
            self.validation_generator.reset()
            preds = self.model.predict_generator(gen_2, steps=self.validation_steps)
            preds_tta.append(preds)
        final_pred = np.mean(preds_tta, axis=0)
        y_pred = np.argmax(final_pred,axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true,y_pred,average='weighted')
        recall = recall_score(y_true,y_pred, average='weighted')
        self.val_f1_scores.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print(f"- val_f1_score: {f1:.5f} - val_precision: {precision:.5f} - val_recall: {recall:.5f}")
        class_report=classification_report(y_true,y_pred,target_names=self.labels,output_dict=True)
        clsf_report = pd.DataFrame(class_report).transpose()
        clsf_report.to_csv(self.save_path+'/epoch-'+str(epoch)+'--classification_report.csv')
        cm = confusion_matrix(y_true,y_pred)
        sns.heatmap(cm,cmap="YlGnBu",xticklabels=self.labels,yticklabels=self.labels,linewidths=2,linecolor='white',annot=True,fmt="d",cbar=False)
        plt.savefig(self.save_path+'/model-epoch-'+str(epoch)+'-result.png')
        cm_as_df=cm2df(cm,self.labels)
        cm_as_df.to_csv(self.save_path+'/epoch-'+str(epoch)+'-confusion_matrix.csv')
        return
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
label_smooh_rate=0.25
Num_epochs=50
backbone="B3_xc_1205_dataset"
batch_size = 32
chnnel_number = 3
root='/share_data/xc_1205_dataset'
img_height = img_width = 256
trainPath = os.path.join(root,'train')
valPath = os.path.join(root,'val')
testPath = '/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1215/test_data/'
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
    shuffle=False,
    class_mode='categorical')
test_generator = val_datagen.flow_from_directory(
    testPath,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=False,
    color_mode='rgb',
    class_mode='categorical')
class_label=['凹陷病变-阿弗他','凹陷病变-溃疡','凹陷病变-糜烂', '肠内容物-寄生虫','肠内容物-血液', '肠内容物-异物',
'隆起病变-结节', '隆起病变-静脉异常','隆起病变-息肉','隆起病变-肿块(肿瘤)', '黏膜-苍白','黏膜-发红', '黏膜-发红2',
'黏膜-颗粒状', '黏膜-水肿（充血）','黏膜-萎缩','黏膜-异常绒毛','平坦病变-白斑', '平坦病变-点',
'平坦病变-红斑', '正常-无效-遮挡','正常-无杂质-翻折发白', '正常-杂质-静脉','正常-无杂质-黏膜红色',
'正常-无杂质-曝光发白','正常-无杂质-绒毛', '正常-杂质-白色固体','正常-杂质-白色漂浮','正常-杂质-黑色肠液', 
'正常-无杂质-挤压发白','正常-无杂质-黑色肠液', '正常-杂质-黄色大块漂浮物', '正常-杂质-黄色小块漂浮物', '正常-杂质-挤压发白',
'正常-无杂质-静脉', '正常-杂质-弥散漂浮物','正常-杂质-曝光发白','正常-杂质-水泡','正常-杂质-LED灯'
]
#val_metrics = Metrics(test_generator,validation_steps=(totalTest // batch_size) + 1,
                      #save_path=weight_path,labels=class_label)

#base_model = load_model('/share_data/code/EfficientNetB3_weights_001_valacc_0.7409_valloss_0.8640.h5')
input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB3(weights='imagenet',input_tensor=input_tensor)
Dense1=Dense(512,activation='relu')(efnet.get_layer('top_dropout').output)
drop1=Dropout(0.25)(Dense1)
Dense2=Dense(1024,activation='relu')(drop1)
drop2=Dropout(0.5)(Dense2)
prediction_output=Dense(class_number,activation='softmax')(drop2)
model = Model(input_tensor,prediction_output)
#model.summary()
#model = multi_gpu_model(ppmodel,gpus=4)
model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy'])

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
#lr_reduce = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,mode='auto')
checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/smr_{}'.format(label_smooh_rate)+'_{epoch:02d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
cbks = [early_stopping,checkpoint,CosineAnnealingScheduler(T_max=50, eta_max=1e-3, eta_min=3e-4)]
def gen_smooth_labels(datagen):
    while True:
        iamge,label = datagen.next()
        yield [iamge,smooth_positive_labels(label,label_smooh_rate)]
histoy=model.fit_generator(gen_smooth_labels(train_generator),epochs=Num_epochs,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=32,
                                   verbose=1,callbacks=cbks)
