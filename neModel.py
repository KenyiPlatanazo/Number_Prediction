from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
#from sklearn.linear_model import LogisticRegression
from sklearn import neighbors #K-Nearest neightbor
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn import metrics
import dill as pickle

print(os.listdir("./input"))

digits = datasets.load_digits()
pd = datasets.load_digits()
print('Digits dictionary content \n{}'.format(digits.keys()))



images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

images_and_labels = list(zip(digits.images, digits.target))
for index, (data, label) in enumerate(images_and_labels[:4]):
    imgdim=int(np.sqrt(digits.data[index].shape[0]))
    img=np.reshape(digits.data[index],(imgdim,imgdim))
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)



X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25)
print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))



#Using LogisticRegression it returns a 96% accuracy
#class_logistic = LogisticRegression()
#class_logistic.fit(X_train, y_train)

#y_pred = class_logistic.predict(X_test)
#print("Accuracy of model = %2f%%" % (accuracy_score(y_test, y_pred )*100))

knn=neighbors.KNeighborsClassifier() #98.666...% accuracy. Way better
knn.fit(X_train, y_train)
y_pred  = knn.predict(X_test)
print("Accuracy of model = %2f%%" % (accuracy_score(y_test, y_pred )*100))

print("Classification report for classifier %s:\n%s\n" % (knn, metrics.classification_report(y_test, y_pred)))

filename = 'knnModel.pk'
with open('./'+filename, 'wb') as file:
    pickle.dump(knn, file) 
