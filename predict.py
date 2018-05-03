import numpy as np
import torch
from net import Net
from PIL import Image
from torchvision import transforms

test_path = '/home/arccha/.kaggle/competitions/digit-recognizer/test.csv'
net = Net()
net.load_state_dict(torch.load('./net_state'))
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()])

X = np.genfromtxt(test_path, delimiter=',', skip_header=1)
X = X.reshape(-1, 28, 28)
X = list(map(Image.fromarray, X))
X = list(map(transform, X))

print('ImageId,Label')
for i, x in enumerate(X):
    y = net(x.unsqueeze(0)).argmax()
    print(str(i + 1) + ',' + str(int(y)))
