from pathlib import Path

from cuda import *
from data import *
from net import *
from torch.utils.data import DataLoader

config_path = Path('train_config.yaml')
config = None
with config_path.open('r') as f:
    config = yaml.load(f)

_, device = get_cuda_if_available()

validation_dataset, _ = train_validation_split(
    config['train_path'], pretransform=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1,
                               shuffle=False, num_workers=1, pin_memory=True)
nets = [CNN(), CNN(), CNN()]
net_dirs = [Path('./' + type(net).__name__ + str(i))
            for i, net in enumerate(nets)]
for net, net_dir in zip(nets, net_dirs):
    net.eval()
    net.to(device)
    net_state_path = net_dir.joinpath('net.state')
    with net_state_path.open(mode='rb') as f:
        net.load_state_dict(torch.load(f))

acc = 0
for x, y_true in tqdm(validation_loader):
    x = x.to(device)
    y_preds = [net(x).argmax(dim=1) for net in nets]
    y_pred = np.argmax(np.bincount(y_preds))
    if y_true.numpy() == y_pred:
        acc += 1
print(acc / 42000.0)
