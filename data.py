import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DigitRecognizerDataset(Dataset):
    def __init__(self, csv_file, max_rows=None):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
        if max_rows:
            data = np.genfromtxt(csv_file, delimiter=',',
                                 skip_header=1, max_rows=max_rows)
        else:
            data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        Y, X = np.split(data, [1], axis=1)
        X = X.reshape(-1, 28, 28)
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, i):
        x = self.X[i]
        x = Image.fromarray(x)
        x = self.transform(x)
        y = int(self.Y[i])
        return (x, y)

    def __len__(self):
        return self.len
