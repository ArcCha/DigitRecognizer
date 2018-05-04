import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import *
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

torch.manual_seed(48)

train_path = '/home/arccha/.kaggle/competitions/digit-recognizer/train.csv'
DATA_NUM = 1000  # 42000 - max
data = np.genfromtxt(train_path, delimiter=',',
                     skip_header=1, max_rows=DATA_NUM)

Y, X = np.split(data, [1], axis=1)
print(X.shape)
X = X.reshape(DATA_NUM, 28, 28)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()])
X = list(map(Image.fromarray, X))
X = list(map(transform, X))

H = {}  # Training history and statistics
net = SimpleCNN()
H['net'] = type(net).__name__
net_dir = Path('./' + H['net'])
net_dir.mkdir(parents=True, exist_ok=True)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
H['optimizer'] = str(optimizer)
criterion = nn.CrossEntropyLoss()
H['criterion'] = str(criterion)

EPOCH_NUM = 10
H['epoch_num'] = EPOCH_NUM
H['loss'] = []
start = time.process_time()
for epoch in tqdm(range(EPOCH_NUM), desc='Total'):
    running_loss = 0.0
    for i, x in enumerate(tqdm(X, desc='Epoch ' + str(epoch))):
        x = x.unsqueeze(0)
        y = torch.LongTensor(Y[i])
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i == 0:
            H['initial_loss'] = loss.item()
        #print("[{}, {}]: {}".format(epoch, i, loss.item()))
    H['loss'].append(running_loss / DATA_NUM)
    #print("Epoch {}: {}".format(epoch, running_loss / DATA_NUM))


end = time.process_time()
H['learning_duration'] = end - start
net_state_path = net_dir.joinpath('net.state')
net_state_path.touch(exist_ok=True)
with net_state_path.open(mode='wb') as f:
    torch.save(net.state_dict(), f)
net_stats_path = net_dir.joinpath('stats.json')
net_stats_path.touch(exist_ok=True)
with net_stats_path.open('w') as f:
    json.dump(H, f, indent=2, sort_keys=True)
