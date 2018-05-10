from pathlib import Path

import Augmentor
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

KAGGLE_PATH = Path.home() / '.kaggle/competitions/digit-recognizer/'
KAGGLE_TEST_PATH = KAGGLE_PATH / 'test.csv'
KAGGLE_TRAIN_PATH = KAGGLE_PATH / 'train.csv'


class DigitRecognizerDataset(Dataset):
    def __init__(self, X, Y, pretransform=False):
        self.pretransform = pretransform
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ToTensor()])
        if self.pretransform:
            X = list(map(self.transform, map(Image.fromarray, X)))
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, i):
        x = self.X[i]
        if not self.pretransform:
            x = Image.fromarray(x)
            x = self.transform(x)
        # If Y is None then assume that we're dealing with test dataset
        if self.Y is not None:
            y = int(self.Y[i])
            return x, y
        return x

    def __len__(self):
        return self.len


def train_validation_split(csv_file, max_rows=None, validation_num=None,
                           pretransform=False):
    if max_rows:
        data = np.genfromtxt(csv_file, delimiter=',',
                             skip_header=1, max_rows=max_rows)
    else:
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    Y, X = np.split(data, [1], axis=1)
    X = X.reshape(-1, 28, 28)
    if validation_num:
        idx = np.random.randint(0, len(X), validation_num)
        V = X[idx]
        X = np.delete(X, idx, axis=0)
        VY = Y[idx]
        Y = np.delete(Y, idx, axis=0)
        validation_dataset = DigitRecognizerDataset(
            V, VY, pretransform=pretransform)
    else:
        validation_dataset = None
    train_dataset = DigitRecognizerDataset(
        X, Y, pretransform=pretransform)
    return train_dataset, validation_dataset


def get_test_dataset(csv_file, pretransform=False):
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    X = data.reshape(-1, 28, 28)
    test_dataset = DigitRecognizerDataset(X, None, pretransform=pretransform)
    return test_dataset


def save_as_png(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    Y, X = np.split(data, [1], axis=1)
    Y = np.squeeze(Y).astype(int)
    root_path = Path('augment/original')
    print(Y.shape)
    root_path.mkdir(parents=True, exist_ok=True)
    class_paths = [root_path.joinpath(str(i)) for i in range(10)]
    for path in class_paths:
        path.mkdir(exist_ok=True)
    idx = [0 for _ in range(10)]
    for i in tqdm(range(len(X))):
        path = class_paths[Y[i]]
        path = path.joinpath(str(idx[Y[i]]).zfill(5) + '.png')
        idx[Y[i]] += 1
        x = X[i].reshape(28, 28)
        image = Image.fromarray(x).convert('RGB')
        with path.open('wb') as f:
            image.save(f, format='PNG')


def augment_data(root_path):
    src_path = root_path.joinpath('original')
    class_paths = [src_path.joinpath(str(i)).resolve() for i in range(10)]
    dst_path = root_path.joinpath('out')
    dst_path.mkdir(exist_ok=True)
    out_paths = [dst_path.joinpath(str(i)) for i in range(10)]
    for path in out_paths:
        path.mkdir(exist_ok=True)
    out_paths = [path.resolve() for path in out_paths]
    p = Augmentor.Pipeline(source_directory=str(
        class_paths[0]), output_directory=str(out_paths[0]), save_format='PNG')
    for i in range(1, 10):
        p.add_further_directory(str(class_paths[i]), str(out_paths[i]))
    p.random_distortion(1.0, 5, 5, 1)
    p.sample(80000)


if __name__ == '__main__':
    config_path = Path('train_config.yaml')
    config = None
    with config_path.open('r') as f:
        config = yaml.load(f)
    # save_as_png(config['train_path'])
    augment_data(Path('augment'))
