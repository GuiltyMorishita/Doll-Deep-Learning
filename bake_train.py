#coding: utf-8

#必要なライブラリのインポート
import cv2
import os
import six
import datetime

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from clf_bake_model import clf_bake

import numpy as np


#その１　------データセット作成------

def getDataSet():
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    brand_list = ['dd_face', 'sd_face', 'pullip_face', 'blythe_face']
    for i, brand in enumerate(brand_list):
        #まずは2値分類を目指すので暦フォルダとothersフォルダの中身だけ引っ張ってきます。
        path = "./doll_image/" #ここにディレクトリのパスを設定
        imgList = os.listdir(path + brand)
        #データを4:1の割合でtrainとtestに分けます。
        imgCount = len(imgList)
        cutCount = imgCount - imgCount / 5
        for j, imgName in enumerate(imgList):
            imgSrc = cv2.imread(path + brand + "/" + imgName)
            #またimreadはゴミを吸い込んでも、エラーで止まらずNoneを返してくれます。
            #ですので読み込み結果がNoneでしたらスキップしてもらいます。
            if imgSrc is None:
                continue

            if j < cutCount:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)

    return X_train,y_train,X_test,y_test



#その３ ---------学習させる-------

def train():
    #上で作った関数でデータセットを用意します。
    X_train,y_train,X_test,y_test = getDataSet()
    #このままだとchainerで読み込んでもらえないので、array型にします。
    X_train = np.array(X_train).astype(np.float32).reshape((len(X_train),3, 50, 50)) / 255
    y_train = np.array(y_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.float32).reshape((len(X_test),3, 50, 50)) / 255
    y_test = np.array(y_test).astype(np.int32)
    # 諸々の初期設定
    model = clf_bake()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    epochNum = 30
    batchNum = 50
    epoch = 1

    # 学習とテスト
    while epoch <= epochNum:
        print("epoch: {}".format(epoch))
        print(datetime.datetime.now())

        trainImgNum = len(y_train)
        testImgNum = len(y_test)

        #---学習---
        sumAcr = 0
        sumLoss = 0

        perm = np.random.permutation(trainImgNum)

        for i in six.moves.range(0, trainImgNum, batchNum):
            #ランダムにbatchNumの数だけ抽出する
            X_batch = X_train[perm[i:i+batchNum]]
            y_batch = y_train[perm[i:i+batchNum]]

            optimizer.zero_grads()
            loss, acc = model.forward(X_batch, y_batch)
            loss.backward()
            optimizer.update()

            sumLoss += float(loss.data) * len(y_batch)
            sumAcr += float(acc.data) * len(y_batch)
        print('train mean loss={}, accuracy={}'.format(sumLoss / trainImgNum, sumAcr / trainImgNum))

        #---テスト---
        sumAcr = 0
        sumLoss = 0

        for i in six.moves.range(0, testImgNum, batchNum):
            X_batch = X_test[i:i+batchNum]
            y_batch = y_test[i:i+batchNum]
            loss, acc = model.forward(X_batch, y_batch, train=False)

            sumLoss += float(loss.data) * len(y_batch)
            sumAcr += float(acc.data) * len(y_batch)
        print('test  mean loss={}, accuracy={}'.format(
            sumLoss / testImgNum, sumAcr / testImgNum))
        epoch += 1
        #モデルの保存
        S.save_hdf5('model'+str(epoch+1), model)

train()
