import numpy as np
from sklearn import tree

from data import KAGGLE_TRAIN_PATH, KAGGLE_TEST_PATH

data = np.genfromtxt(KAGGLE_TRAIN_PATH,
                     delimiter=',', skip_header=1)

Y, X = np.split(data, [1], axis=1)

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X, Y)

X = np.genfromtxt(KAGGLE_TEST_PATH,
                  delimiter=',', skip_header=1)

Y = dtc.predict(X)

print('ImageId,Label')
for i, y in enumerate(Y):
    print(str(i + 1) + ',' + str(int(y)))
