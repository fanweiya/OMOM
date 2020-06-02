from keras.callbacks import Callback
from itertools import tee
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,classification_report
import itertools
from keras.preprocessing.image import ImageDataGenerator,load_img
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf
'''
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
os.system('rm tmp')
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
'''
import math
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname='/root/simhei.ttf',size=14)

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
        if int(epoch)>=50:
            try:
                self.validation_generator.shuffle=False
                # duplicate generator to make sure y_true and y_pred are calculated from the same observations
                #gen_1, gen_2 = tee(self.validation_generator)
                #y_true = (np.vstack(next(gen_1)[1] for _ in range(self.validation_steps))).argmax(axis=1)
                #tta_steps = 3
                #preds_tta = []
                # for i in tqdm(range(tta_steps)):
                    #self.validation_generator.reset()
                    #c,preds = self.model.predict_generator(gen_2, steps=self.validation_steps)
                    #preds_tta.append(preds)
               # final_pred = np.mean(preds_tta, axis=0)
                y_true = self.validation_generator.classes
                gpu='/gpu:%d'%(np.argmax(memory_gpu))
                with tf.device(gpu):
                    final_pred = self.model.predict_generator(self.validation_generator, steps=self.validation_steps)
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
                clsf_report.to_csv(self.save_path+'/epoch-'+str(int(epoch)+1)+'--classification_report.csv')
                cm = confusion_matrix(y_true,y_pred)
                plt.figure(figsize=(20,20))
                sns.set(font=myfont.get_name())
                plt.tight_layout()
                sns.heatmap(cm,cmap="YlGnBu",xticklabels=self.labels,yticklabels=self.labels,linewidths=2,linecolor='white',annot=True, square=True,fmt="d",cbar=False)
                plt.savefig(self.save_path+'/model-epoch-'+str(int(epoch)+1)+'-confusion_matrix-result.png')
                cm_as_df=cm2df(cm,self.labels)
                cm_as_df.to_csv(self.save_path+'/epoch-'+str(int(epoch)+1)+'-confusion_matrix.csv')
                error_index=np.where(False==np.equal(y_true,y_pred))[0].tolist()
                error_sample=np.random.choice(error_index,size=min(len(error_index),30),replace=False).tolist()
                # Display the classification results for the first 30 images
                plt.figure(figsize=(30,30))
                for n in tqdm(error_sample):
                    plt.subplot(math.ceil(len(error_sample)/5),5,error_sample.index(n)+1)
                    plt.imshow(load_img(self.validation_generator.filepaths[n],target_size=(224,224)))
                    predicted_label = self.labels[y_pred[n]]
                    true_label = self.labels[y_true[n]]
                    score=final_pred[n][y_pred[n]]
                    plt.title('{}\n{}:{:.2f}---{}'.format(self.validation_generator.filepaths[n].split(os.sep)[-1],predicted_label,score
                            ,true_label), color='green',fontsize=18)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.save_path+'/model-'+str(int(epoch)+1)+'predictions-error-sample-result.png')
                return
            except:
                self.validation_generator.shuffle=False
                # duplicate generator to make sure y_true and y_pred are calculated from the same observations
                #gen_1, gen_2 = tee(self.validation_generator)
                #y_true = (np.vstack(next(gen_1)[1] for _ in range(self.validation_steps))).argmax(axis=1)
                #tta_steps = 3
                #preds_tta = []
                # for i in tqdm(range(tta_steps)):
                    #self.validation_generator.reset()
                    #c,preds = self.model.predict_generator(gen_2, steps=self.validation_steps)
                    #preds_tta.append(preds)
               # final_pred = np.mean(preds_tta, axis=0)
                y_true = self.validation_generator.classes
                gpu='/gpu:%d'%(np.argmax(memory_gpu))
                with tf.device(gpu):
                    final_pred = self.model.predict_generator(self.validation_generator, steps=self.validation_steps)
                y_pred = np.argmax(final_pred,axis=1)
                f1 = f1_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true,y_pred,average='weighted')
                recall = recall_score(y_true,y_pred, average='weighted')
                self.val_f1_scores.append(f1)
                self.val_recalls.append(recall)
                self.val_precisions.append(precision)
                print(f"- val_f1_score: {f1:.5f} - val_precision: {precision:.5f} - val_recall: {recall:.5f}")
                self.labels=self.validation_generator.class_indices.keys()
                class_report=classification_report(y_true,y_pred,target_names=self.labels
                                                   ,output_dict=True)
                clsf_report = pd.DataFrame(class_report).transpose()
                clsf_report.to_csv(self.save_path+'/epoch-'+str(int(epoch)+1)+'--classification_report.csv')
                cm = confusion_matrix(y_true,y_pred)
                plt.figure(figsize=(20,20))
                sns.set(font=myfont.get_name())
                plt.tight_layout()
                sns.heatmap(cm,cmap="YlGnBu",xticklabels=self.labels,yticklabels=self.labels,linewidths=2,linecolor='white',annot=True, square=True,fmt="d",cbar=False)
                plt.savefig(self.save_path+'/model-epoch-'+str(int(epoch)+1)+'-confusion_matrix-result.png')
                cm_as_df=cm2df(cm,self.labels)
                cm_as_df.to_csv(self.save_path+'/epoch-'+str(int(epoch)+1)+'-confusion_matrix.csv')
                error_index=np.where(np.equal(y_true,y_pred)==False)[0].tolist()
                error_sample=np.random.choice(error_index,size=min(len(error_index),30),replace=False).tolist()
                # Display the classification results for the first 30 images
                plt.figure(figsize=(30,30))
                for n in tqdm(error_sample):
                    plt.subplot(math.ceil(len(error_sample)/5),5,error_sample.index(n)+1)
                    plt.imshow(load_img(self.validation_generator.filepaths[n],target_size=(224,224)))
                    predicted_label = self.labels[y_pred[n]]
                    true_label = self.labels[y_true[n]]
                    score=final_pred[n][y_pred[n]]
                    plt.title('{}\n{}:{:.2f}---{}'.format(self.validation_generator.filepaths[n].split(os.sep)[-1],predicted_label,score
                            ,true_label), color='green',fontsize=18)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.save_path+'/model-'+str(int(epoch)+1)+'predictions-error-sample-result.png')
                return
        else:
            return
if __name__ == '__main__':
    pass
    #val_metrics = Metrics(test_generator,validation_steps=(totalTest // batch_size) + 1,save_path='/root/',labels=['label'])
