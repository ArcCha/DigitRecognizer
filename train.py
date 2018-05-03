import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import Net
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

train_path = '/home/arccha/.kaggle/competitions/digit-recognizer/train.csv'
DATA_NUM = 42000  # 42000 - max
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


net = Net()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    running_loss = 0.0
    for i, x in enumerate(X):
        x = x.unsqueeze(0)
        y = torch.LongTensor(Y[i])
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

torch.save(net.state_dict(), './net_state')
