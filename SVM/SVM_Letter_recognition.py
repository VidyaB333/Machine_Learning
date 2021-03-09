import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\SVM\letter-recognition.csv')
print(df.shape)
col =['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege',
      'xegvy', 'y-ege', 'yegvx']
df.columns = col
x = df[df.columns[1:-1]]
y = df[df.columns[0]]
print(x.shape, x.ndim)
print(y.shape, y.ndim)
print(df.head(4))
print(x.columns)
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
print(type(x_scaler), type(y))
print()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
print(x_train.shape, y_train.shape)


from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=0.1, gamma=5)#gamma= auto is good option
svm = svm.fit(x_train, y_train)


print('Training data performance metrices on liner kernel')
y_train_predict = svm.predict(x_train)
print(accuracy_score(y_train, y_train_predict))
#print(confusion_matrix(y_train, y_train_predict))
#print(classification_report(y_train, y_train_predict))


print('Testing data performance metrices on liner kernel')
y_test_predict = svm.predict(x_test)
print(accuracy_score(y_test, y_test_predict))
#print(confusion_matrix(y_test, y_test_predict))
#print(classification_report(y_test, y_test_predict))
print(svm.support_vectors_)
print(type(svm.support_vectors_))
print(svm.support_vectors_.shape)
op = svm.predict(np.array([5,12,3,7,2,10,5,5,4,13,3,9,2,8,10]).reshape(1,-1))
print(op)