import random
import numpy as np
def generate_data():
  data, lbls = [], []
  for _ in range(50000):
    lbl = random.random() > 0.5
    if lbl:
      slab = random.randint(0,2)
      mean_y = slab * 2 - 2
      mean_x = 1
      data.append([(random.random() - 0.5)*1.6 + mean_x, random.random()*0.4-0.2 + mean_y])
      lbls.append(1)
    else:
      slab = random.randint(0,1)
      mean_y = slab * 2 - 1
      mean_x = -1
      data.append([(random.random() - 0.5)*1.6 + mean_x, random.random()*0.4-0.2 + mean_y])
      lbls.append(0)
  return np.array(data, dtype=np.float32), np.array(lbls, dtype=np.float32)