import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import Sequence, to_categorical
import numpy as np
from random import sample, randint, shuffle
import glob
import cv2
import time
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
class DataLoader_video_train(Sequence):
    def __init__(self, path1, batch_size = 4):
        self.batch_size = batch_size
        self.path = '/data/stars/user/rdai/charades/charades_SSD/'
        #self.path = '/dev/shm/full_body_charades/'
        csv_file= pd.read_csv(path1)
        nan_list=[]
        for i in range(len(csv_file['actions'])):
            if pd.isnull(csv_file['actions'][i])==1:
                nan_list.extend([i])
        csv_file=csv_file.drop(index=nan_list)
        self.files = [i.strip() for i in csv_file['id']]
        #print('files')
        #print(self.files[:50])
        self.annotations = [i for i in csv_file['actions']]
        self.stack_size = 64
        self.num_classes = 157
        self.stride = 2
        #self.label=[('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156')]
        self.label=[tuple([str(i) for i in range(0, 157)])]

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        #print(idx)
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_train = [self._get_video(i) for i in (batch)]
        x_train = np.array(x_train, np.float32)
        #normalization the image value from -1 to 1
        x_train /= 127.5
        x_train -= 1
        #changes
        batch1= self.annotations[idx * self.batch_size : (idx + 1) * self.batch_size]
        action_list = np.array([])
        mlb = MultiLabelBinarizer()
        mlb.fit(self.label)
        mlb.transform([tuple(action_list)])
        counter=0
        y_train=np.array([])
        for i in (batch1):
            action_list_1=[]
            counter = counter + 1
            list1=i.split(';')
            length1=len(list1)
            for k in range(length1):
                list2=list1[k].split(' ')
                #length2 = len(list2)
                action=int(list2[0][-3:])
                action_list_1.extend([str(action)])
            if counter==1:
                y_train = mlb.transform([tuple(action_list_1)])
            else:
                y_train=np.vstack((y_train, mlb.transform([tuple(action_list_1)])))
        #print y_train
        return x_train, y_train

    def _get_video(self, vid_name):
        #print(vid_name)
        images = glob.glob(self.path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])
        #print arr
        return arr

class DataLoader_video_train_old2(Sequence):
    def __init__(self, path1, batch_size = 4):
        self.batch_size = batch_size
        self.path = '/data/stars/user/rdai/charades/charades_SSD/'
        csv_file= pd.read_csv(path1)
        nan_list=[]
        for i in range(len(csv_file['actions'])):
            if pd.isnull(csv_file['actions'][i])==1:
                nan_list.extend([i])
        csv_file=csv_file.drop(index=nan_list)
        self.files = [i.strip() for i in csv_file['id']]
        #print('files')
        #print(self.files[:50])
        self.annotations = [i for i in csv_file['actions']]
        self.stack_size = 64
        self.num_classes = 157
        self.stride = 2

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        print(idx)
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        #print('batch')
        #print(batch)
        x_train = [self._get_video(i) for i in (batch)]
        x_train = np.array(x_train, np.float32)
        #normalization the image value from -1 to 1
        x_train /= 127.5
        x_train -= 1
        #changes
        batch1= self.annotations[idx * self.batch_size : (idx + 1) * self.batch_size]
        action_list = np.array([])
        counter=0
        for i in (batch1):
            action_list_1=[]
            counter = counter + 1
            list1=i.split(';')
            length1=len(list1)
            for k in range(length1):
                list2=list1[k].split(' ')
                #length2 = len(list2)
                action=int(list2[0][-3:])
                #action_arr=np.array(action)
                #action_cat=to_categorical(action_arr, self.num_classes)
                action_list_1.extend([action])
                action_list_2.array(action_list_1)
                action_list_2=np.unique(action_list)
            for j in len(action_list_2):
                action_arr = np.array(j)
                action_cat = to_categorical(action_arr, self.num_classes)
                if k==0:
                    action_list=action_cat
                else:
                    action_list=action_list+action_cat
            if counter==1:
                y_train = action_list
            else:
                y_train=np.vstack((y_train, action_list))

        print y_train
        return x_train, y_train

    def _get_video(self, vid_name):
        #print(vid_name)
        images = glob.glob(self.path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])
        print arr
        return arr

class DataLoader_video_train_old(Sequence):
    def __init__(self, path1, batch_size = 4):
        self.batch_size = batch_size
        self.path = '/data/stars/user/rdai/charades/charades_SSD/'
        csv_file= pd.read_csv(path1)
        nan_list=[]
        for i in range(len(csv_file['actions'])):
            if pd.isnull(csv_file['actions'][i])==1:
                nan_list.extend([i])
        csv_file=csv_file.drop(index=nan_list)
        self.files = [i.strip() for i in csv_file['id']]
        #print('files')
        #print(self.files[:50])
        self.annotations = [i for i in csv_file['actions']]
        self.stack_size = 64
        self.num_classes = 157
        self.stride = 2

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        print(idx)
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        #print('batch')
        #print(batch)
        x_train = [self._get_video(i) for i in (batch)]
        x_train = np.array(x_train, np.float32)
        #normalization the image value from -1 to 1
        x_train /= 127.5
        x_train -= 1
        #changes
        batch1= self.annotations[idx * self.batch_size : (idx + 1) * self.batch_size]
        action_list = np.array([])
        counter=0
        for i in (batch1):
            counter = counter + 1
            list1=i.split(';')
            length1=len(list1)
            for k in range(length1):
                list2=list1[k].split(' ')
                #length2 = len(list2)
                action=int(list2[0][-3:])
                action_arr=np.array(action)
                action_cat=to_categorical(action_arr, self.num_classes)
                if k==0:
                    action_list=action_cat
                else:
                    action_list=action_list+action_cat
            if counter==1:
                y_train = action_list
            else:
                y_train=np.vstack((y_train, action_list))

        print y_train
        return x_train, y_train

    def _get_video(self, vid_name):
        #print(vid_name)
        images = glob.glob(self.path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])
        print arr
        return arr
