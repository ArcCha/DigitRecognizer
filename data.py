import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DigitRecognizerDataset(Dataset):
    def __init__(self, X, Y, pretransform=False, test=False):
        self.test = test
        self.pretransform = pretransform
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
        if self.pretransform:
            X = list(map(self.transform, map(Image.fromarray, X)))
        self.X = X
        if not test:
            self.Y = Y
        self.len = len(X)

    def __getitem__(self, i):
        x = self.X[i]
        if not self.pretransform:
            x = Image.fromarray(x)
            x = self.transform(x)
        if not self.test:
            y = int(self.Y[i])
            return (x, y)
        return x

    def __len__(self):
        return self.len


def train_validation_split(csv_file, max_rows=None, validation_num=None, test=False):
    if max_rows:
        data = np.genfromtxt(csv_file, delimiter=',',
                             skip_header=1, max_rows=max_rows)
    else:
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    if not test:
        Y, X = np.split(data, [1], axis=1)
    else:
        X = data
        Y = None
    X = X.reshape(-1, 28, 28)
    if validation_num:
        idx = np.random.randint(0, len(X), validation_num)
        V = X[idx]
        X = np.delete(X, idx, axis=0)
        VY = Y[idx]
        Y = np.delete(Y, idx, axis=0)
        validation_dataset = DigitRecognizerDataset(V, VY)
    else:
        validation_dataset = None
    train_dataset = DigitRecognizerDataset(X, Y, test=test)
    return (train_dataset, validation_dataset)
