from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')

image= mnist.data.to_numpy()
plt.subplot(431)
plt.imshow((image[0].reshape(28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(432)
plt.imshow(image[1].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(433)
plt.imshow(image[3].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(434)
plt.imshow(image[4].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(435)
plt.imshow(image[5].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(436)
plt.imshow(image[6].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')

index_number= np.random.permutation(70000)
x1,y1=mnist.data.loc[index_number],mnist.target.loc[index_number]
x1.reset_index(drop=True,inplace=True)
y1.reset_index(drop=True,inplace=True)
x_train , x_test = x1[:55000], x1[55000:]
y_train , y_test = y1[:55000], y1[55000:]

svc = svm.SVC(gamma='scale',class_weight='balanced',C=100)
svc.fit(x_train,y_train)
result=svc.predict(x_test)

print('Accuracy :',accuracy_score(y_test,result))
print(classification_report(y_test,result))

knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(x_train, y_train)
result=knn.predict(x_test)

print('Accuracy :',accuracy_score(y_test,result))
print(classification_report(y_test,result))
