"""
    isort:skip_file
"""
import itertools

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt


def imshow(img):
    plt.imshow(img, cmap='gray')


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          filesave=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    plt.figure(figsize=(7, 7))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    yticks = []
    for i in (range(cm.shape[0])):
        acc = cm[i, i] / np.sum(cm[i])
        yticks.append('{} (acc={:.10f})'.format(i, acc))

    plt.yticks(tick_marks, yticks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if filesave is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(filesave, bbox_inches='tight')
