import numpy as np
from sklearn import tree

data = np.genfromtxt('/home/arccha/.kaggle/competitions/digit-recognizer/train.csv',
                     delimiter=',', skip_header=1)

Y, X = np.split(data, [1], axis=1)

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X, Y)

X = np.genfromtxt('/home/arccha/.kaggle/competitions/digit-recognizer/test.csv',
                  delimiter=',', skip_header=1)

Y = dtc.predict(X)

print('ImageId,Label')
for i, y in enumerate(Y):
    print(str(i + 1) + ',' + str(int(y)))
