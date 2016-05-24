#coding: utf-8

import os

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S

import numpy as np

#chainerからChainを継承してクラスを作成します。
class clf_bake(chainer.Chain):
    def __init__(self):

        super(clf_bake, self).__init__(
            # ※1. このネットワークの部分は後々改善させます。
            conv1 =  F.Convolution2D(3, 16, 5, pad=2),
            conv2 =  F.Convolution2D(16, 32, 5, pad=2),
            l3    =  F.Linear(6272, 256),
            l4    =  F.Linear(256, 4) #最終的に2クラスに分類させるため出力は2
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, X_data, y_data, train=True):
        self.clear() #初期化
        X_data = chainer.Variable(np.asarray(X_data), volatile=not train)
        y_data = chainer.Variable(np.asarray(y_data), volatile=not train)
        h = F.max_pooling_2d(F.relu(self.conv1(X_data)), ksize = 5, stride = 2, pad =2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize = 5, stride = 2, pad =2)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)
        return F.softmax_cross_entropy(y, y_data), F.accuracy(y, y_data)
