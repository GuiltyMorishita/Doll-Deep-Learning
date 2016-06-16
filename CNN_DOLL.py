#! -*- coding: utf-8 -*-

import time
import six.moves.cPickle as pickle
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, Chain, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix


class ImageNet(Chain):
    def __init__(self, n_outputs):
        super(ImageNet, self).__init__(
            conv1 =  L.Convolution2D(3, 32, 5), #padding=0, stride=1 filtersize=5
            conv2 =  L.Convolution2D(32, 32, 5),

            # l3 =     L.Linear(512, 512), #32
            # l3 =     L.Linear(1568, 512), #50
            l3 =     L.Linear(2592, 512), #64
            l4 =     L.Linear(512, n_outputs)
        )

    def forward(self, x_data, y_data, train=True, gpu=-1):

        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # padding=0
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        # h = F.spatial_pyramid_pooling_2d(F.relu(self.conv2(h)), 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)


        if train == False: # 評価時にのみ以下を実行
            cnt = 0
            missid = []

            for ydata in y.data:
                # ファイル出力して確認するなら
                # fp_.write(str(np.argmax(ydata)))
                # fp_.write(' ')

                if y_data[cnt] != np.argmax(ydata):
                    # 識別に失敗したデータを出力する処理．
                    missid.append(glob_z_test[z_batch[cnt]])

                cnt += 1

            glob_all_missid.extend(missid)
                # 全バッチにおいて識別失敗した id を格納

        return F.softmax_cross_entropy(y, t), F.accuracy(y,t)


    def predict(self, x_test, gpu=-1):
        if gpu >= 0:
            x_test = cuda.to_gpu(x_test)
        x = Variable(x_test)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # padding=0
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        # h = F.spatial_pyramid_pooling_2d(F.relu(self.conv2(h)), 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.l3(h)))
        y = self.l4(h)
        predictions = np.array([], np.float32)
        for o in y.data:
            predictions = np.append(predictions, np.array([np.argmax(o)], np.float32))
        return predictions


class CNN:
    def __init__(self, data, target, n_outputs, gpu=-1):

        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        file_ids = range(len(target))
        global glob_z_test

        self.x_train,\
        self.x_test,\
        self.y_train,\
        self.y_test,\
        self.z_train,\
        glob_z_test = train_test_split(data, target, file_ids, test_size=0.1)

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def predict(self, x_test, gpu=-1):
        return self.model.predict(x_test, gpu)


    def train_and_test(self, n_epoch=100, batchsize=100):

        epoch = 1
        best_accuracy = 0
        log = ""



        while epoch <= n_epoch:
            print 'epoch', epoch
            log = log + 'epoch ' + str(epoch) + '\n'

            perm = np.random.permutation(self.n_train)
            sum_train_accuracy = 0
            sum_train_loss = 0

            global glob_all_missid
            glob_all_missid = []

            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.x_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]
                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            train_mean = 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)
            print train_mean
            log = log + train_mean + '\n'

            # evaluation
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.x_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                global z_batch
                z_batch = range(i, i+batchsize)

                real_batchsize = len(x_batch)

                loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            test_mean = 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
            print test_mean
            log = log + test_mean + '\n'

            epoch += 1

        yh = self.predict(self.x_test,gpu=1)
        confmat = confusion_matrix(self.y_test, yh)
        print confmat
        log = log + str(confmat)

        d = datetime.datetime.today()

        log_filename = "log_64x64_2592_" + d.strftime("%Y-%m-%d_%H%M%S") + ".txt"
        with open(log_filename, "w") as f:
            f.write(log)

        serializers.save_hdf5('doll_model', self.model)


    # def dump_model(self):
    #     self.model.to_cpu()
    #     pickle.dump(self.model, open(self.model_name, 'wb'), -1)
    #
    # def load_model(self):
    #     self.model = pickle.load(open(self.model_name,'rb'))
    #     if self.gpu >= 0:
    #         self.model.to_gpu()
    #     self.optimizer.setup(self.model.collect_parameters())
