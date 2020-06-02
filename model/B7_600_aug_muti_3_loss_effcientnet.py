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
import functools
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'
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
lr=3e-4
Num_epochs=50
backbone="EfficientNetB7_300muti3label"
batch_size = 16
chnnel_number = 3
root='/share_data/xc_1205_dataset'
img_height = img_width = 300
trainPath = os.path.join(root,'train')
valPath = os.path.join(root,'val')
testPath = os.path.join(root,'test')
class_number = len(os.listdir(trainPath))
weight_path='/root/CEST_weights/{}'.format(backbone)
if not os.path.exists(weight_path):
    os.makedirs(weight_path) 
# determine the total number of image paths in training validation
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))
if root.split(os.sep)[-1]=='xc_1205_dataset':
    img_mean = np.array([118.39764148472882, 85.12477094174385, 55.27967261855539], dtype="float32")
    print(img_mean)
if root.split(os.sep)[-1]=='V7.0_20190927_src_expand':
    img_mean = np.array([99.52549092226955, 81.8520933362902, 70.2714537825208], dtype="float32")
    print(img_mean)        
train_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
    rescale=1. / 255,
    preprocessing_function=random_augimage()
    )
val_datagen = ImageDataGenerator(
    rescale=1. / 255)
test_datagen = ImageDataGenerator(
    rescale=1. / 255)
train_datagen.mean = img_mean
val_datagen.mean = img_mean   
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
test_generator = test_datagen.flow_from_directory(
    testPath,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=False,
    color_mode='rgb',
    class_mode='categorical')
from itertools import tee  # finally! I found something useful for it
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,classification_report
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams,font_manager
from sklearn.metrics import confusion_matrix
rcParams['font.size'] = 6
def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.grid(False)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    img_time=time.strftime('%H:%M:%S',time.localtime(time.time()))
    plt.savefig(weight_path+'/model-{}-time-{}.png'.format(str(params['root'].split(os.sep)[-1]),str(img_time)))
    plt.close()
        #plt.show()
class Metrics(Callback):
    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
    def on_train_begin(self, logs={}):
        self.val_f1_scores = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        # duplicate generator to make sure y_true and y_pred are calculated from the same observations
        gen_1, gen_2 = tee(self.validation_generator)
        y_true = np.vstack(next(gen_1)[1] for _ in range(self.validation_steps)).astype('int')
        y_pred = (self.model.predict_generator(gen_2, steps=self.validation_steps)).astype('int')
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        self.val_f1_scores.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        report = classification_report(y_true.argmax(axis=1),y_pred.argmax(axis=1),target_names=self.validation_generator.class_indices.keys())
        cm = confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
        print(f" - val_f1_score: {f1:.5f} - val_precision: {precision:.5f} - val_recall: {recall:.5f}")
        print(report)
        plot_confusion_matrix(cm,classes=self.validation_generator.class_indices.keys())
        return
val_metrics = Metrics(test_generator,validation_steps=(totalTest // batch_size) + 1)
input_tensor = Input((img_height,img_width,chnnel_number))
efnet=EfficientNetB7(weights='imagenet',input_tensor=input_tensor)
prediction_output=Dense(class_number,activation='softmax')(efnet.get_layer('top_dropout').output)
predicton_masterlabel_output=Dense(2,activation='sigmoid',name="sigmoid")(Flatten()(efnet.get_layer('block3b_add').output))
predicton_master3label_output=Dense(6,activation='softmax',name="softmax2")(Flatten()(efnet.get_layer('block6b_add').output))
model = Model(efnet.inputs,[prediction_output,predicton_masterlabel_output,predicton_master3label_output])
#model.summary()
#model = multi_gpu_model(ppmodel,gpus=4)
#lr_normalizer(params['lr'],params['optimizer']))
model.compile(loss=['categorical_crossentropy','binary_crossentropy','categorical_crossentropy'],loss_weights=[1,1,1],optimizer=optimizers.Adam(lr=lr),metrics=['accuracy',top3_acc,top5_acc])
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_dense_1_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
if root.split(os.sep)[-1]=='xc_1205_dataset':
    early_stopping = EarlyStopping(monitor='val_dense_1_loss',patience=10,verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='val_dense_1_loss',patience=5,verbose=1,mode='auto')
    checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/weights_{}'.format(label_smooh_rate)+'_{epoch:03d}_valacc_{val_dense_1_acc:.4f}_valloss_{val_dense_1_loss:.4f}.h5',monitor='val_dense_1_loss',verbose=1)
# mc = ModelCheckpoint('/content/drive/My Drive/biCNNicres_checkpoint.h5'monitor='val_acc' save_best_only=True verbose=1)
cbks = [early_stopping,lr_reduce,checkpoint]
def multi_3label(number):
    masterlabel={}
    for i,value in enumerate([i.split('---')[number] for i in train_generator.class_indices.keys()]):
        for j,d_value in enumerate(list(set([i.split('---')[number] for i in train_generator.class_indices.keys()]))):
            if value==d_value:
                masterlabel[i] = j
    return masterlabel
def multi_label(number):
    masterlabel={}
    for i,value in enumerate([i.split('---')[number] for i in train_generator.class_indices.keys()]):
            if value=='zheng-chang':
                masterlabel[i] = 1
            else:
                masterlabel[i] = 0
    return masterlabel
def gen_data_labels(datagen):
    while True:
        iamge,labels = datagen.next()
        #yield [iamge,smooth_positive_labels(label,params['label_smooh_rate'])]
        master_labels = np.zeros((labels.shape[0],2)).astype("float32") ##master laebel zhengcahng or yichang
        master_3labels = np.zeros((labels.shape[0],6)).astype("float32") ##master laebel 6
        for i in range(master_labels.shape[0]):
            master_labels[i][multi_label(0)[np.argmax(labels[i])]] = 1.0
        for i in range(master_labels.shape[0]):
            master_3labels[i][multi_3label(0)[np.argmax(labels[i])]] = 1.0
        yield [iamge,[smooth_positive_labels(labels,label_smooh_rate),master_labels,master_3labels]]
histoy=model.fit_generator(gen_data_labels(train_generator),epochs=Num_epochs,
                               steps_per_epoch=totalTrain // batch_size,
                               validation_data=gen_data_labels(validation_generator),
                               validation_steps=totalVal // batch_size,
                               verbose=1,callbacks=cbks)
