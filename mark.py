#! -*- coding: utf-8 -*-
import csv
import re
from sklearn.metrics import confusion_matrix
import dalmatian

# fileList = ["watanabe.csv", "katui.csv", "saitou.csv"]

csvfile = "human2.csv"
f = open(csvfile, "r")
reader = csv.reader(f)

cnt = 0
answer = 0

y_true = []
y_pred = []
brandCntDict = {}
labels = ["blythe", "dollfiedream", "pullip", "sahra", "superdollfie", "xcute"]
# labels = ["A", "B", "C", "D"]
for row in reader:
    matchOB = re.search(r'\\dataset\\(brand|dealer)\\(\w+)\\', row[0])
    # matchOB = re.search(r'\\dataset\\brand\\(\w+)\\', row[0])
    true = matchOB.group(2)
    pred = row[4]

    print(true, pred)

    y_true.append(true)
    y_pred.append(pred)

    if true == pred:
        answer += 1
    cnt += 1

    if true in brandCntDict:
        brandCntDict[true] += 1
    else:
        brandCntDict[true] = 1


for k, v in sorted(brandCntDict.items(), key=lambda x:x[1]):
    print(k, v)


print(cnt, answer)
print(float(answer)/cnt)

data = confusion_matrix(y_true, y_pred)
print(data)
mx = dalmatian.Matrix(labels, data)

#Options
# mx.cell_size = 10.0 #[mm]
# mx.font_size = 14
# mx.label_font_size = 7
mx.cell_color = "black" #black, red, yellow, green, blue, purple
mx.label_color = "black" #black, white
mx.line_type = "normal" #normal, dot
mx.percentage = True
mx.draw()


f.close()
