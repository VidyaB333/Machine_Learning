import numpy as np
import pandas as pd

x = np.random.randint(10,30,20)
for i in range(len(x)):
    print(x[i])
y = np.random.randint(10,30,20)
for i in range(len(y)):
    print(y[i])

from matplotlib import pyplot as plt
plt.scatter(x, y)


#Standardization
mu = np.mean(x)
sd = np.std(x)
print(mu, sd)
x_Normalize = (x-mu)/sd
print(x_Normalize)


mu_y = np.mean(y)
sd_y = np.std(y)
print(mu_y, sd_y)
y_Normalize = (y-mu_y)/sd_y
print(y_Normalize)
plt.scatter(x_Normalize, y_Normalize, color ='g')
plt.show()




#Normalization
min =np.min(x)
max = np.max(x)
x_standard = (x-min)/(max-min)
print(x_standard)

min_y =np.min(y)
max_y = np.max(y)
y_standard = (y-min_y)/(max_y-min_y)
print(y_standard)
plt.scatter(x_standard, y_standard, color ='b')
plt.show()