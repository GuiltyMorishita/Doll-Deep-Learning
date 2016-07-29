#! -*- coding: utf-8 -*-

from CNN_DOLL import CNN
from dollface import DollFaceDataset
from chainer import cuda

#GPUつかうよ
cuda.init(0)
missCountDict = {}
for i in range(0, 100):
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

    missIdList = cnn.train_and_test(n_epoch=100)

    for missId in missIdList:
        if dataset.index2filename[missId] in missCountDict:
            missCountDict[dataset.index2filename[missId]] += 1
        else:
            missCountDict[dataset.index2filename[missId]] = 1


print(dataset.index2name)
for k, v in sorted(missCountDict.items(), key=lambda x:x[1]):
    print(k, v)
