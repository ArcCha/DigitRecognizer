from torch.utils.data import DataLoader
from tqdm import tqdm

from cuda import *
from data import *
from net import *

if not KAGGLE_TEST_PATH.exists():
    KAGGLE_TEST_PATH = '../test.csv'
else:
    KAGGLE_TEST_PATH = str(KAGGLE_TEST_PATH)

test_dataset = get_test_dataset(KAGGLE_TEST_PATH)
test_loader = DataLoader(dataset=test_dataset, batch_size=32,
                         shuffle=False, num_workers=1, pin_memory=True)

_, device = get_cuda_if_available()
net = SimpleCNN()
net_dir = Path('./' + type(net).__name__)
net.to(device)
net_state_path = net_dir.joinpath('net.state')
with net_state_path.open(mode='rb') as f:
    net.load_state_dict(torch.load(f))

result_path = net_dir.joinpath('result.txt')
with result_path.open('w') as f:
    f.write('ImageId,Label\n')
    for i, x in enumerate(tqdm(test_loader)):
        x = x.to(device)
        y = net(x).argmax(dim=1)
        f.write(str(i + 1) + ',' + str(int(y)) + '\n')
