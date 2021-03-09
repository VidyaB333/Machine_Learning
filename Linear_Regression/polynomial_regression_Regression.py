import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
x = np.array(x)
y = np.array(y)
#Polynomial Regression
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x.reshape(-1,1)) #It added the extra featurtes with higher degrees in input
print(x_poly)

model_poly = LinearRegression()
model_poly.fit(x_poly, y)
plt.scatter(x, y)
plt.plot(x, model_poly.predict(x_poly))

r2 = mean_squared_error(y, model_poly.predict(x_poly) )
print('R squared with polynomail regression: ', r2)


model_linear = LinearRegression()
model_linear.fit(x.reshape(-1,1),y)
r2_linear = mean_squared_error(y, model_linear.predict((x).reshape(-1,1)) )
print('R squared with Liear regression: ', r2_linear)
plt.plot(x, model_linear.predict(np.array(x).reshape(-1,1)))
plt.show()