#! -*- coding: utf-8 -*-

from CNN_DOLL import CNN
from dollface import DollFaceDataset
from chainer import cuda

#GPUつかうよ
cuda.init(0)

print 'load DollFace dataset'
dataset = DollFaceDataset()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.get_n_types_target()

cnn = CNN(data = data,
          target = target,
          gpu = 0,
          n_outputs = n_outputs)

cnn.train_and_test(n_epoch = 100)
