from cuda import *
from data import *
from net import *
from torch.utils.data import DataLoader
from tqdm import tqdm

if not KAGGLE_TEST_PATH.exists():
    KAGGLE_TEST_PATH = '../test.csv'
else:
    KAGGLE_TEST_PATH = str(KAGGLE_TEST_PATH)

test_dataset = get_test_dataset(KAGGLE_TEST_PATH)
test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                         shuffle=False, num_workers=1, pin_memory=True)

_, device = get_cuda_if_available()
nets = [CNN(), CNN(), CNN()]
net_dirs = [Path('./' + type(net).__name__ + str(i))
            for i, net in enumerate(nets)]

for net, net_dir in zip(nets, net_dirs):
    net.eval()
    net.to(device)
    net_state_path = net_dir.joinpath('net.state')
    with net_state_path.open(mode='rb') as f:
        net.load_state_dict(torch.load(f))

result_path = Path('result.txt')
with result_path.open('w') as f:
    f.write('ImageId,Label\n')
    for i, x in enumerate(tqdm(test_loader)):
        x = x.to(device)
        ys = [net(x).argmax(dim=1) for net in nets]
        y = np.argmax(np.bincount(ys))
        f.write(str(i + 1) + ',' + str(int(y)) + '\n')
