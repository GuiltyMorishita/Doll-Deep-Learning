


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
