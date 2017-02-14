#! -*- coding: utf-8 -*-

from CNN_DOLL import CNN
from ALEX_DOLL import ALEX
from dollface import DollFaceDataset
from chainer import cuda
import dalmatian
from tqdm import tqdm

#GPUつかうよ
cuda.init(0)
missCountDict = {}
confmats = []
for i in tqdm(range(0, 10)):
    print 'load DollFace dataset'
    dataset = DollFaceDataset()
    dataset.load_data_target()
    data = dataset.data
    target = dataset.target
    n_outputs = dataset.get_n_types_target()

    cnn = ALEX(data = data,
              target = target,
              gpu = 0,
              n_outputs = n_outputs)

    missIdList, confmat = cnn.train_and_test(n_epoch=100)

    for missId in missIdList:
        if dataset.index2filename[missId] in missCountDict:
            missCountDict[dataset.index2filename[missId]] += 1
        else:
            missCountDict[dataset.index2filename[missId]] = 1

    confmats.append(confmat)

print(dataset.index2name)
for k, v in sorted(missCountDict.items(), key=lambda x:x[1]):
    print(k, v)

totalConfmat = reduce(lambda x,y: x+y, confmats)
print totalConfmat
labels = ["blythe", "dollfiedream", "pullip", "sahra", "superdollfie", "xcute"]
# labels = ["A", "B", "C", "D"]
mx = dalmatian.Matrix(labels, totalConfmat)

#Options
# mx.cell_size = 10.0 #[mm]
# mx.font_size = 14
# mx.label_font_size = 7
mx.cell_color = "black" #black, red, yellow, green, blue, purple
mx.label_color = "black" #black, white
mx.line_type = "normal" #normal, dot
mx.percentage = True
mx.draw()
