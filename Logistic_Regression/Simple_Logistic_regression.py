import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\User_Data_purchase.csv')
print(df.shape)
print(df.columns)

#sns.pairplot(df)
#plt.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

sns.countplot(df['Purchased'])
plt.show()

sns.boxplot(df['Purchased'], df['Age'])
plt.show()

sns.displot(df['Age'])
plt.show()
X = df[df.columns[2:-1]]
y= df[df.columns[-1]]
print(X.shape, y.shape)
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)
print(X_train[0:10][:])
print(type(X_train))
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
print(type(X_train))
print(X_train[0:10, : ])
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
print(model.coef_)
print(model.intercept_)