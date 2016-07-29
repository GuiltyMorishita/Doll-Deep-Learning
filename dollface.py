#! -*- coding: utf-8 -*-

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv
import re
import shutil

class DollFaceDataset:
    def __init__(self):
        self.data_dir_path = u"./dealer_image/"
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u'doll_dataset'
        self.image_size = 50
        self.index2filename = {}

    def get_dir_list(self):
        tmp = os.listdir(self.data_dir_path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])

    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def load_data_target(self):
        if os.path.exists(self.dump_name):
            self.load_dataset()

        if self.target is None:
            dir_list = self.get_dir_list()
            self.target = []
            target_name = []
            self.data = []

            #カスケードファイル読み込み
            cascade_path = "./lbpcascade_animeface/lbpcascade_animeface.xml"
            if not os.path.isfile(cascade_path):
                raise RuntimeError("%s: not found" % cascade_path)
            cascade = cv.CascadeClassifier(cascade_path)

            if os.path.exists("./dataset_image"):
                shutil.rmtree("./dataset_image")

            cnt = 0
            for dir_name in dir_list:
                print(dir_name)
                os.makedirs("./dataset_image" + "/" + dir_name)
                file_list = os.listdir(self.data_dir_path+dir_name)
                pattern = re.compile(r'.*\.(jpg|jpeg|png)$', re.IGNORECASE)
                for file_name in file_list:
                    matchOB = re.match(pattern, file_name)
                    abs_name = self.data_dir_path+dir_name+'/'+file_name
                    if matchOB is None:
                        print('"' + abs_name + '" is not image file.')
                        continue
                    # read class id i.e., target
                    class_id = self.get_class_id(abs_name)
                    # 画像データの読み込み
                    image = cv.imread(abs_name)
                    #グレースケール変換
                    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    gray = cv.equalizeHist(gray)

                    #顔認識実行
                    facerect = cascade.detectMultiScale(gray,
                        # detector options
                        scaleFactor = 1.15, # dealer
                        # scaleFactor = 1.10, # brand
                        minNeighbors = 5,
                        minSize = (self.image_size, self.image_size))

                    for (i, rect) in enumerate(facerect):
                        print(i)
                        #顔だけ切り出し
                        x = rect[0]
                        y = rect[1]
                        width = rect[2]
                        height = rect[3]
                        image = image[y:y+height, x:x+width]
                        if image.shape[0] < self.image_size or image.shape[1] < self.image_size:
                            continue
                        image = cv.resize(image, (self.image_size, self.image_size))
                        root, ext = os.path.splitext(file_name)
                        new_image_path = "./dataset_image" + "/" + dir_name + "/" + root + "_" + str(i) + ext
                        cv.imwrite(new_image_path, image)
                        image = image.transpose(2,0,1)
                        image = image/255.

                        self.data.append(image)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))

                        self.index2filename[cnt] = dir_name + "/" + root + "_" + str(i) + ext
                        cnt += 1

            print(cnt)
            self.index2name = {}
            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i]

        self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        self.dump_dataset()

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name,self.index2filename), open(self.dump_name, 'wb'), -1)

    def load_dataset(self):
        self.data, self.target, self.index2name, self.index2filename = pickle.load(open(self.dump_name, 'rb'))

# dataset = DollFaceDataset()
# dataset.load_data_target()
