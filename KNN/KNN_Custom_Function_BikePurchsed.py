import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt


def Standardization(X):
    print('In Standardization method')
    #print(X[0:10])
    mu = np.mean(X)
    sd = np.std(X)
    #print(mu, sd)

    for i in range(len(X)):
        X[i] = round((X[i]-mu)/sd, 4)
    return X

#Indices of neighbors
def find_neighbors(X,y , new, k):
    indices = []
    lst = np.zeros(y.shape)
    a = np.array(X['Age'])
    b = np.array(X['EstimatedSalary'])
    step1 = (a-new[0])**2
    step2 = (b-new[1])**2
    lst = (np.sqrt(step1 + step2)).reshape(-1,1)
    indicator = (np.array(X.index)).reshape(-1,1)

    temp = list(lst)
    #print(lst)
    #print(indices)
    #print(type(lst), lst.shape)
    #print(type(indices), indices.shape)
    data = np.concatenate([lst, indicator], axis=1)
    #print(data)

    for i in range(k):
        index_of_min_distance = np.argmin(temp)
        index_in_training_set = data[index_of_min_distance][1]
        indices.append(index_in_training_set)
        #print(index)
        temp.pop(index_of_min_distance)
    return indices

#Labels of given indices
def find_labels(neighbors, y, new, index):
    #print('In method for finding labels')

    #print(neighbors)
    labels = []

    for i in neighbors:
        if i in index:
            #print('Yes')
            labels.append(y[i])

    #print('Label of neighbors: ', labels)
    return labels

#Ways to find out labels
def Criteria_to_find_label(labels):
    dist ={}
    for i in labels:
        if i in dist:
            dist[i] += 1
        else:
            dist[i] =1
    #print(dist)
    no_of_keys = []
    for keys, values in dist.items():
        no_of_keys.append(keys)


    mx = dist[no_of_keys[0]]
    if len(no_of_keys)>=2:
        mx = max(dist[0], dist[1])

    for key, value in dist.items():
        if value == mx:
            print('Predicted label :', key)
            return key


#Total diatance to check the different in actual and predicted value
def hammingDist(str1, str2):
    i = 0
    count = 0

    while (i < len(str1)):
        if (str1[i] != str2[i]):
            count += 1
        i += 1
    return count

def plotting(X, Y):
    print('IN plotting')
    df = pd.concat([X, Y], axis=1)
    print(df)
    y0 = df[df['Purchased'] == 1]
    y1 = df[df['Purchased'] == 0]

    print(y0)

    print(y1)

    plt.scatter(y0['Age'], y0['EstimatedSalary'], color='blue')

    plt.scatter(y1['Age'], y1['EstimatedSalary'], color='red')



if __name__ =='__main__':
    df = pd.read_csv(
        'C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\User_Data_purchase.csv')
    X = df[['Age', 'EstimatedSalary']]
    Y= df['Purchased']

    #Changing the datatypes of series to float
    X = X.astype({'Age':'float', 'EstimatedSalary':'float'})

    #Standardization of datapoints
    X['Age'] = Standardization(X['Age'])
    X['EstimatedSalary'] =  Standardization(X['EstimatedSalary'])

    #Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
    plotting(X_train, y_train)

    #Different values of k to tune the model
    List_of_K=[1,3,5,7,9,11]
    List_of_difference = []

    for k in List_of_K:
        print('Value of k:', k)
        train_index = X_train.index
        train_index = list(train_index)
        # print(train_index)
        print(X_test)
        op = []
        for i in range(len(X_test)):
            print('***********')
            temp = X_test.iloc[i, :]

            # print(temp)
            new_data = list(temp)
            # print(temp)
            print('New test data: {} and label :{}'.format(new_data, y_test.iloc[i]))
            plt.scatter(new_data[0], new_data[1], s=10, color='green')

            indices = find_neighbors(X_train, Y, new_data, k)
            labels = find_labels(indices, Y, new_data, train_index)
            output = Criteria_to_find_label(labels)
            op.append(output)

        print('Predicted labels for x_test: ')
        t = pd.Series(op, index=[X_test.index])
        t.columns = ['Predicted_Labels']

        print(y_test.count(), y_test.sum())
        print(t.count(), t.sum())

        #For haming distance calculation
        t = list(t)
        y = list(y_test)
        m = ''
        n = ''
        for i in t:
            m = m + str(i)

        for i in y:
            n = n + str(i)
        distance = hammingDist(m, n)
        List_of_difference.append(distance)


    plt.show()

    for i in range(len(List_of_K)):
        print('For {}, distance: {}'.format(List_of_K[i], List_of_difference[i]))










