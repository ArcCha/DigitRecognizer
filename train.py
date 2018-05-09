import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from cuda import *
from data import *
from net import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

config_path = Path('train_config.yaml')
config = None
with config_path.open('r') as f:
    config = yaml.load(f)
H = {}  # Training history and statistics
CUDA, device = get_cuda_if_available()
H['cuda'] = CUDA

train_path = config['train_path']
H['data_num'] = config['data_num']
H['validation_num'] = config['validation_num']
H['batch_size'] = config['batch_size']
train_dataset, validation_dataset = train_validation_split(
    train_path, max_rows=config['data_num'], validation_num=config['validation_num'], pretransform=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

validation_classes = np.zeros(10)
for x, y in tqdm(validation_loader, desc='Validation stats'):
    validation_classes[y] += 1
H['validation_classes'] = validation_classes.tolist()

net = SimpleCNN()
H['net'] = type(net).__name__
net_dir = Path('./' + H['net'])
net_dir.mkdir(parents=True, exist_ok=True)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
H['optimizer'] = str(optimizer)
criterion = nn.CrossEntropyLoss()
H['criterion'] = str(criterion)

H['epoch_num'] = config['epoch_num']
H['loss'] = []
H['train_acc'] = []
H['test_acc'] = []
start = time.process_time()
for epoch in tqdm(range(config['epoch_num']), desc='Total'):
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc='Epoch ' + str(epoch)):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    H['loss'].append(
        running_loss / (config['data_num'] / config['batch_size']))
    acc = 0
    for x, y_true in tqdm(train_loader, desc='Train acc ' + str(epoch)):
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = net(x).argmax(dim=1)
        acc += y_true.eq(y_pred).sum()
    acc = float(acc) / (len(train_loader) * config['batch_size'])
    H['train_acc'].append(acc)
    acc = 0
    for x, y_true in tqdm(validation_loader, desc='Validation ' + str(epoch)):
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = net(x).argmax(dim=1)
        acc += y_true.eq(y_pred).sum()
    acc = float(acc) / config['validation_num']
    H['test_acc'].append(acc)
    net_state_path = net_dir.joinpath('net' + str(epoch) + '.state')
    net_state_path.touch(exist_ok=True)
    with net_state_path.open(mode='wb') as f:
        torch.save(net.state_dict(), f)
    if epoch % 100 == 99:
        net_stats_path = net_dir.joinpath('stats' + str(epoch) + '.json')
        net_stats_path.touch(exist_ok=True)
        with net_stats_path.open('w') as f:
            json.dump(H, f, indent=2, sort_keys=True)

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
