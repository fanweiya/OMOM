import talos as ta
from talos.utils import lr_normalizer
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

# root='/home/server1/Desktop/fanweiya/diease_20190822_expand'

def smooth_positive_labels(inputs,smooth_factor):
    if 0 <= smooth_factor<= 1:
        K = inputs.shape[-1]
        smooth_lable=((1 - smooth_factor) * inputs) + (smooth_factor / K)
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))        
    return smooth_lable

p = {'label_smooh_rate':(0,0.4,4),
         'optimizer': [optimizers.Adam],
         'lr': [3e-4],
         'Num_epochs': [50],
         'root':['/share_data/xc_1205_dataset'],
         'backbone':[EfficientNetB0]
#               'backbone':[EfficientNetB0,EfficientNetB1,EfficientNetB2,
#                      EfficientNetB3,EfficientNetB4,EfficientNetB5,
#                      EfficientNetB6,EfficientNetB7]
       }
def EfficientNet(x,y,x_val,y_val,params):
    batch_size = 32
    chnnel_number = 3
    root=params['root']
    #efnet=params['backbone']()	
    efnet=load_model('/share_data/code/weights_0.0_024_valacc_0.9012_valloss_0.3316.h5')
    model=efne
    img_height = img_width = efnet.get_layer(index=0).output_shape[1]
    trainPath = os.path.join(root,'train')
    valPath = os.path.join(root,'val')
    testPath = os.path.join(root,'test')
    class_number = len(os.listdir(trainPath))
    weight_path='/root/CEST_weights/{}'.format(params['backbone'])
    if not os.path.exists(weight_path):
        os.makedirs(weight_path) 
    # determine the total number of image paths in training validation
    # and testing directories
    totalTrain = len(list(paths.list_images(trainPath)))
    totalVal = len(list(paths.list_images(valPath)))
    totalTest = len(list(paths.list_images(testPath)))
    if params['root'].split(os.sep)[-1]=='xc_1205_dataset':
        img_mean = np.array([118.39764148472882, 85.12477094174385, 55.27967261855539], dtype="float32")
        print(img_mean)
    if params['root'].split(os.sep)[-1]=='V7.0_20190927_src_expand':
        img_mean = np.array([99.52549092226955, 81.8520933362902, 70.2714537825208], dtype="float32")
        print(img_mean)        
    train_datagen = ImageDataGenerator(
        rotation_range=np.random.uniform(0,360),
        fill_mode='constant',
        cval=0,
        rescale=1. / 255,
        #shear_range=np.random.uniform(0,10),
        #width_shift_range=np.random.uniform(0,0.1),#hide x
        #height_shift_range=np.random.uniform(0,0.1),#hide y
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.9,1.3],
        brightness_range=[0.5,1.5],#channel_shift_range=np.random.uniform(0,5)
        )
    val_datagen = ImageDataGenerator(
        rescale=1. / 255
)
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
)
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
        def __init__(self, validation_generator, validation_steps, threshold=0.5):
            self.validation_generator = validation_generator
            self.validation_steps = validation_steps
            self.threshold = threshold
        def on_train_begin(self, logs={}):
            self.val_f1_scores = []
            self.val_recalls = []
            self.val_precisions = []
        def on_epoch_end(self, epoch, logs={}):
            # duplicate generator to make sure y_true and y_pred are calculated from the same observations
            gen_1, gen_2 = tee(self.validation_generator)
            y_true = np.vstack(next(gen_1)[1] for _ in range(self.validation_steps)).astype('int')
            y_pred = (self.model.predict_generator(gen_2, steps=self.validation_steps)>self.threshold).astype('int')
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
    #input_tenser = Input((img_height,img_width,chnnel_number))
    #prediction_output=Dense(class_number,activation='softmax')(efnet.get_layer('top_dropout').output)
    #model = Model(efnet.inputs,prediction_output)
    #model.summary()
    #model = multi_gpu_model(ppmodel,gpus=4)
    lr = params['lr']
    m_epoch=params['Num_epochs']
    #lr_normalizer(params['lr'],params['optimizer']))
    model.compile(loss='categorical_crossentropy',optimizer=params['optimizer'](lr=lr),metrics=['accuracy'])
    class ParallelModelCheckpoint(ModelCheckpoint):
        def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=1):
            self.single_model = model
            super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
        def set_model(self, model):
            super(ParallelModelCheckpoint,self).set_model(self.single_model)
    if params['root'].split(os.sep)[-1]=='xc_1205_dataset':
        print(params['root'])
        early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,mode='auto')
        checkpoint = ParallelModelCheckpoint(model,filepath=weight_path+'/weights_{}'.format(params['label_smooh_rate'])+'_{epoch:03d}_valacc_{val_acc:.4f}_valloss_{val_loss:.4f}.h5',monitor='val_loss',verbose=1)
    # mc = ModelCheckpoint('/content/drive/My Drive/biCNNicres_checkpoint.h5'monitor='val_acc' save_best_only=True verbose=1)
    cbks = [early_stopping,lr_reduce,checkpoint,val_metrics]
    def gen_smooth_labels(datagen):
        while True:
            iamge,label = datagen.next()
            yield [iamge,smooth_positive_labels(label,params['label_smooh_rate'])]
    histoy=model.fit_generator(gen_smooth_labels(train_generator),epochs=m_epoch,
                                   steps_per_epoch=totalTrain // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=totalVal // batch_size,
                                   use_multiprocessing = True,workers=32,
                                   verbose=1,callbacks=cbks)
    return histoy,model
if __name__=='__main__':
    dummy_x = np.empty((1,1,1))
    dummy_y = np.empty((1,1,1))
    scan_results=ta.Scan(x=dummy_x,y=dummy_y,params=p,model=EfficientNet,experiment_name="efficientnet",disable_progress_bar=True,print_params=True)
    scan_results.data.head()
