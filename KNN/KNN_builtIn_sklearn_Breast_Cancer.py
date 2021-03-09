import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\KNN\\Cancer_data.csv')
df.drop('id', axis=1, inplace=True) #Removed Unwanted column
print(df.shape)

#plt.show()
#df['diagnosis'] = df[ df[df['diagnosis']=='M'] =1 and  df[df['diagnosis']=='B'] =0]
number = LabelEncoder()
df['diagnosis'] = number.fit_transform(df['diagnosis'].astype('str'))

print(df['diagnosis'].count(), df['diagnosis'].sum())
#print(df['diagnosis'])

print(df.shape)

corre_coef = df.corr()
sns.heatmap(corre_coef, annot=True)
plt.show()
#sns.heatmap(corre_coef)
#plt.show()

#print(df.isna().count())
y = df['diagnosis']

df.drop(columns=['diagnosis', 'Unnamed: 32'], axis=1, inplace=True)
print(df.shape)

print(y.shape)
x_scaler = StandardScaler().fit_transform(df.values)

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, train_size=0.8, random_state=12)

KNN= KNeighborsClassifier( n_neighbors=10, p=2, metric='minkowski')
KNN.fit(x_train, y_train)


print('Train confusion')
#print(pd.crosstab(y_train, KNN.predict(x_train), rownames='actual', colnames='Predicted'))
y_train_pred = KNN.predict(x_train)
cm = confusion_matrix(y_train, y_train_pred )
print(cm)
print(classification_report(y_train, y_train_pred))
print(round(accuracy_score(y_train, y_train_pred),4))

print('Performance metices on Testing dataset')
y_test_pred = KNN.predict(x_test)
cm = confusion_matrix(y_test, y_test_pred )
print(cm)
print(classification_report(y_test, y_test_pred))
print(round(accuracy_score(y_test, y_test_pred),4))



#Tuning the hyperparameter in KNN
k_arr = [1,2,3,4,5, 6, 7, 8,9,10,11,12,13, 14,15,16,17,18,19,20]
dummy_array = np.zeros((20,3))
df = pd.DataFrame(dummy_array)
df.columns=['k value', 'Training accuracy', 'Testing accuray']


for i in range(len(k_arr)):
    #p=1 as our feature vector contains values for different types of measures-->manhattan distance
    #p=2,when feature vector contain same type of measures lik e ht, wd-->euclidean distance
    KNN = KNeighborsClassifier(k_arr[i],weights='uniform', p=2, metric='minkowski')
    KNN.fit(x_train, y_train)
    train_predict = KNN.predict(x_train)
    tr = accuracy_score(y_train, train_predict)
    test_predict = KNN.predict(x_test)
    te =accuracy_score(y_test, test_predict)
    df.iloc[i, 0] = k_arr[i]
    df.iloc[i, 1] = round(tr, 4)
    df.iloc[i, 2] = round(te, 4)
    print(df.iloc[i, :])

print(df)
print()
plt.plot(np.arange(0,20), df['Training accuracy'],color = 'm', label='Training Accuracy')
plt.plot(np.arange(0,20), df['Testing accuray'], color = 'g', label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing accuracy with different values of K')
plt.xlabel('K Values')
plt.ylabel('Model accuracies')
plt.show()