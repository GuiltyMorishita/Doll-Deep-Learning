


import numpy as np
import dalmatian

# labels = ["blythe", "dollfiedream", "pullip", "sahra", "superdollfie", "xcute"]
labels = ["A", "B", "C", "D"]
# data = np.array([
#   [85, 5, 10, 0, 0, 0],
#   [5, 59, 12, 0, 18, 6],
#   [0, 3, 94, 0, 3, 0],
#   [0, 0, 0, 77, 0, 23],
#   [0, 5, 6, 0, 83, 6],
#   [0, 8, 0, 14, 0, 86],
#   ])

[[1219   40  284    2   20  123]
 [  47  988  151  125  108  177]
 [ 209  174 2155   26   90  129]
 [  27  166   20 1129   14  552]
 [  11  340  137   30 1034   65]
 [  34  107   35  335   24 2673]]

data = np.array([
  [66, 0, 26, 8],
  [0, 100, 0, 0],
  [10, 50, 40, 0],
  [0, 0, 25, 75],
  ])

mx = dalmatian.Matrix(labels, data)

#Options
mx.cell_color = "black" #black, red, yellow, green, blue, purple
mx.label_color = "black" #black, white
mx.line_type = "normal" #normal, dot
mx.percentage = True
mx.draw()
