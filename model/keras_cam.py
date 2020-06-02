import efficientnet.keras as efn 
import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from imutils import paths
from keras import optimizers
import tensorflow as tf
from keras import losses
import random
import os
from keras import regularizers
import numpy as np_utils
from keras.engine.topology import Layer
from keras.models import load_model
from keras import initializers
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams,font_manager
from sklearn.metrics import confusion_matrix
from keras.layers import Dense,Flatten,Input,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
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
from itertools import tee  # finally! I found something useful for it
import itertools
import shutil
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import  partial
import multiprocessing
import glob
import random
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
from IPython.display import HTML
pd.set_option('display.max_colwidth', -1)
def move_file(parpams,path):
    x=parpams[0]
    y=parpams[1]
    result=x.split(os.sep)[-2]+'_result3_2'
    result_path=os.path.join(path,result)
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except:
            pass
    re_path=os.path.join(result_path,y)
    if not os.path.isdir(re_path):
        try:
            os.makedirs(re_path)
        except:
            pass
    else:
        name=x.split(os.sep)[-1]
        file_path=os.path.join(re_path,name)
        shutil.copy2(x,file_path)
        file_path
def move_file_score(parpams,path):
    x=parpams[0]
    y=parpams[1]
    result=x.split(os.sep)[-2]+'_result3_2'
    result_path=os.path.join(path,result)
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except:
            pass
    re_path=os.path.join(result_path,y[0])
    if not os.path.isdir(re_path):
        try:
            os.makedirs(re_path)
        except:
            pass
    else:
        name='score_'+'%.2f'%(y[1])+'_'+x.split(os.sep)[-1]
        file_path=os.path.join(re_path,name)
        shutil.copy2(x,file_path)
def move_file_throed_score(parpams,path,throed):
    x=parpams[0]
    y=parpams[1]
    result=x.split(os.sep)[-2]+'_result3_4'
    result_path=os.path.join(path,result)
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except:
            pass
    re_path=os.path.join(result_path,y[0])
    if not os.path.isdir(re_path):
        try:
            os.makedirs(re_path)
        except:
            pass
    elif y[1]>throed:
        name=x.split(os.sep)[-1]
        file_path=os.path.join(re_path,name)
        shutil.copy2(x,file_path)
def anysis_result_output(parpams,label,result):
    filepath=parpams[0]
    class_prob=parpams[1]
    argsort_1=(-class_prob).argsort()[0]
    argsort_2=(-class_prob).argsort()[1]
    argsort_3=(-class_prob).argsort()[2]
    argsort_4=(-class_prob).argsort()[3]
    argsort_5=(-class_prob).argsort()[4]
    result.append(
     {'image_name':filepath.split(os.sep)[-1],
      'image_path':str(filepath),
      'image_view':filepath,
      'Heatmap':'none',
      'top1/score':str(label[argsort_1])+'---'+'%.2f'%(class_prob[argsort_1]),
      'top2/score':str(label[argsort_2])+'---'+'%.2f'%(class_prob[argsort_2]),
      'top3/score':str(label[argsort_3])+'---'+'%.2f'%(class_prob[argsort_3]),
      'top4/score':str(label[argsort_4])+'---'+'%.2f'%(class_prob[argsort_4]),
      'top5/score':str(label[argsort_5])+'---'+'%.2f'%(class_prob[argsort_5])
      },ignore_index=True)
    return result
def createFigure(i):
    fig, ax = plt.subplots(figsize=(.4,.4))
    fig.subplots_adjust(0,0,1,1)
    ax.axis("off")
    ax.axis([0,1,0,1])
    c = plt.Circle((.5,.5), .4, color=cmap(i))
    ax.add_patch(c)
    ax.text(.5,.5, str(i), ha="center", va="center")
    return fig

def fig2inlinehtml(fig,i):
    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0) 
    figdata_png = base64.b64encode(figfile.getvalue())
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)
    return imgstr

def mapping(i):
    fig = createFigure(i)
    return fig2inlinehtml(fig,i)
    
def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i
def image_base64(im):
#     if isinstance(im, str):
#         im = get_thumbnail(im)
    im = Image.open(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()
def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

batch_size = 512
img_height = 224
img_width = 224
chnnel_number = 3
testPath = '/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1205/test_result/分类模型结果'
campath='/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1205/test_result/cam_result'
totalTest = len(list(paths.list_images(testPath)))
test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            #horizontal_flip=np.random.choice((True,False)),
            #vertical_flip=np.random.choice((True,False))
    )
test_generator = test_datagen.flow_from_directory(
        testPath,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        shuffle=False,
        color_mode='rgb')
print('kaishi')
class_label=['正常-无杂质-曝光发白','正常-无杂质-翻折发白','正常-无杂质-挤压发白',
     '正常-无杂质-黑色肠液', '正常-无杂质-静脉','正常-无杂质-黏膜红色', '正常-无杂质-绒毛', '正常-杂质-白色固体',
     '正常-杂质-曝光发白', '正常-杂质-黑色肠液', '正常-杂质-挤压发白', '正常-杂质-静脉','正常-杂质-LED灯','正常-杂质-白色漂浮',
     '正常-杂质-黄色大块漂浮物', '正常-杂质-弥散漂浮物', '正常-杂质-黄色小块漂浮物', '正常-杂质-水泡', '正常-杂质-镜头异物',
    '正常-无效-遮挡',
     '凹陷病变-阿弗他','凹陷病变-溃疡',
     '凹陷病变-糜烂', '内容物-钩虫',
     '肠内容物-血液', '肠内容物-异物',
     '隆起病变-结节','隆起病变-静脉异常','隆起病变-息肉',
     '隆起病变-肿块(肿瘤)', '黏膜-苍白',
     '黏膜-发红','黏膜-发红2','黏膜-颗粒状', '黏膜-水肿（充血）',
     '黏膜-萎缩','黏膜-异常绒毛',
     '平坦病变-白斑', '平坦病变-点']
model=load_model('/root/CEST_weights/39/new/new_B7_weights_017_valacc_0.9522_valloss_0.2120.h5',compile=False)
final_pred = model.predict_generator(generator=test_generator, steps=(totalTest // batch_size) + 1)
result_anysis=pd.DataFrame(columns=['image_name','image_path','image_view','Heatmap','top1/score',
             'top2/score','top3/score','top4/score','top5/score'])
campath='/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1205/test_result/cam_result'
if not os.path.exists(campath):
        os.makedirs(campath)
for filepath,class_prob in zip(test_generator.filepaths,final_pred):
    filename=filepath.split(os.sep)[-1]
    cmap_path=os.path.join(campath,filename)
    if not os.path.exists(cmap_path):
        img = image.load_img(filepath, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        class_idx = np.argmax(preds[0])
        class_output = model.output[:, class_idx]
        last_conv_layer = model.get_layer("block7d_add")
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(128):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(filepath)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        cv2.imencode('.jpg', superimposed_img)[1].tofile(cmap_path)
    result=pd.DataFrame(data={'image_name':filepath.split(os.sep)[-1],
          'image_path':filepath,
          'image_view':filepath,
          'Heatmap':cmap_path,
          'top1/score':class_label[(-class_prob).argsort()[0]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[0]]),
          'top2/score':class_label[(-class_prob).argsort()[1]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[1]]),
          'top3/score':class_label[(-class_prob).argsort()[2]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[2]]),
          'top4/score':class_label[(-class_prob).argsort()[3]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[3]]),
          'top5/score':class_label[(-class_prob).argsort()[4]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[4]])
          },index=[0])
    #print(result)
    result_anysis=result_anysis.append(result,ignore_index=True)
#HTML(result_anysis.to_html(formatters={'image_view': image_formatter}, escape=False))
with pd.option_context('display.max_colwidth', -1):
    result_anysis.to_html('/root/result.html',formatters={'image_view': image_formatter,'Heatmap':image_formatter}, escape=False)
result_anysis1=pd.DataFrame(columns=['image_name','image_path','image_view','Heatmap','top1/score',
             'top2/score','top3/score','top4/score','top5/score'])
campath1='/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1205/test_result/cam_result1'
if not os.path.exists(campath1):
        os.makedirs(campath1)
for filepath,class_prob in zip(test_generator.filepaths,final_pred):
    img = image.load_img(filepath, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    cls = np.argmax(model.predict(x))
    y_c = input_model.output[0, cls]
    #cost = ??????????????????cost*label_index??y_c??????
    conv_output = model.get_layer("top_conv").output
    #conv_output = target_conv_layer, mixed10??????1,5,5,2048
    grads = K.gradients(y_c, conv_output)[0]
    #grads = normalize(grads)
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,img_height/cam.shape[0])
    heatmap = cam / np.max(cam) # scale 0 to 1.0    
    img = cv2.imread(filepath)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    filename=filepath.split(os.sep)[-1]
    cmap_path=os.path.join(campath1,filename)
    cv2.imencode('.jpg', superimposed_img)[1].tofile(cmap_path)
    result=pd.DataFrame(data={'image_name':filepath.split(os.sep)[-1],
          'image_path':filepath,
          'image_view':filepath,
          'Heatmap':cmap_path,
          'top1/score':class_label[(-class_prob).argsort()[0]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[0]]),
          'top2/score':class_label[(-class_prob).argsort()[1]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[1]]),
          'top3/score':class_label[(-class_prob).argsort()[2]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[2]]),
          'top4/score':class_label[(-class_prob).argsort()[3]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[3]]),
          'top5/score':class_label[(-class_prob).argsort()[4]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[4]])
          },index=[0])
    #print(result)
    result_anysis1=result_anysis1.append(result,ignore_index=True)
#HTML(result_anysis.to_html(formatters={'image_view': image_formatter}, escape=False))
with pd.option_context('display.max_colwidth', -1):
    result_anysis1.to_html('/root/result1.html',formatters={'image_view': image_formatter,'Heatmap':image_formatter}, escape=False)
result_anysis2=pd.DataFrame(columns=['image_name','image_path','image_view','Heatmap','top1/score',
             'top2/score','top3/score','top4/score','top5/score'])
campath2='/share_data/小肠/测试集/测试集CEST分类20191125/testdata_1205/test_result/cam_result2'
if not os.path.exists(campath2):
        os.makedirs(campath2)
for filepath,class_prob in zip(test_generator.filepaths,final_pred):
    img = image.load_img(filepath, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    cls = np.argmax(model.predict(x))
    y_c = input_model.output[0, cls]
    #cost = ??????????????????cost*label_index??y_c??????
    conv_output = model.get_layer("block7d_add").output
    #conv_output = target_conv_layer, mixed10??????1,5,5,2048
    grads = K.gradients(y_c, conv_output)[0]
    #grads = normalize(grads)
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,img_height/cam.shape[0])
    heatmap = cam / np.max(cam) # scale 0 to 1.0    
    img = cv2.imread(filepath)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    filename=filepath.split(os.sep)[-1]
    cmap_path=os.path.join(campath2,filename)
    cv2.imencode('.jpg', superimposed_img)[1].tofile(cmap_path)
    result=pd.DataFrame(data={'image_name':filepath.split(os.sep)[-1],
          'image_path':filepath,
          'image_view':filepath,
          'Heatmap':cmap_path,
          'top1/score':class_label[(-class_prob).argsort()[0]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[0]]),
          'top2/score':class_label[(-class_prob).argsort()[1]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[1]]),
          'top3/score':class_label[(-class_prob).argsort()[2]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[2]]),
          'top4/score':class_label[(-class_prob).argsort()[3]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[3]]),
          'top5/score':class_label[(-class_prob).argsort()[4]]+'--'+'%.2f'%(class_prob[(-class_prob).argsort()[4]])
          },index=[0])
    #print(result)
    result_anysis2=result_anysis2.append(result,ignore_index=True)
#HTML(result_anysis.to_html(formatters={'image_view': image_formatter}, escape=False))
with pd.option_context('display.max_colwidth', -1):
    result_anysis2.to_html('/root/result2.html',formatters={'image_view': image_formatter,'Heatmap':image_formatter}, escape=False)

