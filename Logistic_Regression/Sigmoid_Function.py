import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
x = []
Y = []


for i in range(-100,100):
    x.append(i)
    y  = 1/(1 + math.exp(-i))
    Y.append(y)

plt.scatter(x, Y)
plt.show()