#coding:utf-8

import numpy as np
import six
import cv2
import os

import chainer
from chainer import computational_graph as c
import chainer.functions as F
import chainer.serializers as S
from chainer import optimizers

from clf_bake_model import clf_bake

#モデルの読み込み
model = clf_bake()
S.load_hdf5('model22', model)

#キャラクターの名前
chara_name = ['dd', "sd", "pullip", "blythe"]

#伝播の設定
def forward(x_data):
    x = chainer.Variable(x_data, volatile=False)
    h = F.max_pooling_2d(F.relu(model.conv1(x)), ksize = 5, stride = 2, pad =2)
    h = F.max_pooling_2d(F.relu(model.conv2(h)), ksize = 5, stride = 2, pad =2)
    h = F.dropout(F.relu(model.l3(h)), train=False)
    y = model.l4(h)

    return y

#顔検出関数
def detect(image, cascade_file = "./lbpcascade_animeface/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    print(faces)

    return faces

#検出された顔を識別する関数
def recognition(image, faces):
    face_images = []

    for (x, y, w, h) in faces:
        dst = image[y:y+h, x:x+w]
        dst = cv2.resize(dst, (50, 50))
        face_images.append(dst)

    face_images = np.array(face_images).astype(np.float32).reshape((len(face_images),3, 50, 50)) / 255

    #face_images = cuda.to_gpu(face_images)

    return forward(face_images) , image

#識別結果を描画する関数
def draw_result(image, faces, result):



    count = 0
    for (x, y, w, h) in faces:
        result_data = result.data[count]
        classNum = result_data.argmax()
        recognized_class = chara_name[result_data.argmax()]
        if classNum == 0:
            #cv2.putText(image, recognized_class, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,255), 3)
        elif classNum == 1:
            #cv2.putText(image, recognized_class, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 3)
        elif classNum == 2:
            #cv2.putText(image, recognized_class, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,255), 3)
        elif classNum == 3:
            #cv2.putText(image, recognized_class, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0), 3)

        count+=1

    return image

#ファイル読み込み
img = cv2.imread("test2.jpg")

faces = detect(img)

result, image = recognition(img, faces)
print(result.data.argmax())

image = draw_result(image, faces, result)
cv2.imwrite('out.png',image)
