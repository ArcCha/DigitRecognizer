import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data import *
from net import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

H = {}  # Training history and statistics


torch.manual_seed(48)

train_path = '/home/arccha/.kaggle/competitions/digit-recognizer/train.csv'
DATA_NUM = 1000  # 42000 - max
H['data_num'] = DATA_NUM
BATCH_SIZE = 5
H['batch_size'] = BATCH_SIZE
train_dataset = DigitRecognizerDataset(train_path, max_rows=DATA_NUM)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)

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
    for x, y in tqdm(train_loader, desc='Epoch ' + str(epoch)):
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    H['loss'].append(running_loss / BATCH_SIZE)


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
