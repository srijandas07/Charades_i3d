import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, multi_gpu_model

import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from Charades_Loader_test import *

epochs = int(sys.argv[1])
model_name = sys.argv[2]
#version = sys.argv[3]
num_classes = 157
batch_size = 4
stack_size = 64

class i3d_modified:
    def __init__(self, weights = 'rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top = True, weights= weights)
        
    def i3d_flattened(self, num_classes = 157):
        i3d = Model(inputs = self.model.input, outputs = self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        new_model  = Model(inputs = i3d.input, outputs = predictions)
        
        #for layer in i3d.layers:
        #    layer.trainable = False
        
        return new_model

class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


i3d = i3d_modified(weights = 'rgb_imagenet_and_kinetics')
model = i3d.i3d_flattened(num_classes = num_classes)
optim = SGD(lr = 0.01, momentum = 0.9)

#model = load_model("../weights3/epoch11.hdf5")
# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 10)
#filepath = '../weights3/weights.{epoch:04d}-{val_loss:.2f}.hdf5'
csvlogger = CSVLogger('i3d_'+model_name+'.csv')

model.load_weights('./weights_0810_100_epoc/epoch_49.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

model_checkpoint = CustomModelCheckpoint(model, './weights_'+model_name+'/epoch_')
#model_checkpoint = ModelCheckpoint('./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

train_generator = DataLoader_video_train('/data/stars/share/charades/Charades/Charades_v1_train.csv', batch_size = batch_size)
val_generator = DataLoader_video_train('/data/stars/share/charades/Charades/Charades_v1_test.csv',  batch_size = batch_size)
test_generator = DataLoader_video_test('/data/stars/share/charades/Charades/Charades_v1_test.csv',  batch_size = batch_size)
test_generator_with_no_action = DataLoader_video_test_with_no_action('/data/stars/share/charades/Charades/Charades_v1_test.csv',  batch_size = batch_size)
print 'Start testing...'
print 'with_no_action'
prediction_result_with_no_action = parallel_model.predict_generator(generator = test_generator_with_no_action)
np.save("/data/stars/user/rdai/charades/I3D_charades/result/"+str(model_name)+"_with_no_action.txt" , prediction_result_with_no_action)
print 'without_no_action'
prediction_result = parallel_model.predict_generator(generator = test_generator)
np.save("/data/stars/user/rdai/charades/I3D_charades/result/"+str(model_name)+".txt",prediction_result)

import pandas as pd
import numpy as np
#print(parallel_model.evaluate_generator(generator = test_generator))
csv_file= pd.read_csv('/data/stars/share/charades/Charades/Charades_v1_test.csv')
id_video = [i for i in csv_file['id']]
id_video = id_video[:len(prediction_result_with_no_action)/10]
x_test = np.array([])
counter_1 = 0
for j in range(len(prediction_result_with_no_action)/10):
    counter_1 = counter_1 + 1
    x_test1 = prediction_result_with_no_action[10*j:10 * (j + 1)].max(axis=0)
    #x_test1=prediction_result[:10*(j+1)].mean(axis=0)
    if counter_1 == 1:
        x_test = x_test1
    else:
        x_test = np.vstack((x_test, x_test1))

for k in range(x_test.shape[0]):
    with open('out_put_for_matlab'+str(model_name)+'_with_no_action.txt', 'a+') as w:
        w.write(str(id_video[k])+' ')
        for i in range(x_test.shape[1]):
            w.write(str(x_test[k][i])+' ')
        w.write('\n'+'\n')

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
#print(parallel_model.evaluate_generator(generator = test_generator))
csv_file= pd.read_csv('/data/stars/share/charades/Charades/Charades_v1_test.csv')
nan_list = []
for i in range(len(csv_file['actions'])):
    if pd.isnull(csv_file['actions'][i]) == 1:
        nan_list.extend([i])
csv_file = csv_file.drop(index=nan_list)
id_video = [i for i in csv_file['id']]
id_video = id_video[:len(prediction_result)/10]
x_test = np.array([])
counter_1 = 0
for j in range(len(prediction_result)/10):
    counter_1 = counter_1 + 1
    x_test1 = prediction_result[10*j:10 * (j + 1)].max(axis=0)
    #x_test1=prediction_result[:10*(j+1)].mean(axis=0)
    if counter_1 == 1:
        x_test = x_test1
    else:
        x_test = np.vstack((x_test, x_test1))

for k in range(x_test.shape[0]):
    with open('out_put_for_matlab'+str(model_name)+'.txt', 'a+') as w:
        w.write(str(id_video[k])+' ')
        for i in range(x_test.shape[1]):
            w.write(str(x_test[k][i])+' ')
        w.write('\n'+'\n')