from pathlib import Path

from data import *
from net import *
from torch.utils.data import DataLoader

config_path = Path('train_config.yaml')
config = None
with config_path.open('r') as f:
    config = yaml.load(f)

validation_dataset, _ = train_validation_split(
    config['train_path'], pretransform=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1,
                               shuffle=False, num_workers=1, pin_memory=True)
nets = [CNN(), CNN(), CNN()]
net_dirs = [Path('./' + type(net).__name__ + str(i))
            for i, net in enumerate(nets)]

_, device = get_cuda_if_available()

for x, y_true in tqdm(validation_loader):
    x = x.to(device)
    y_true = y_true.to(device)
    y_preds = [net(x).argmax(dim=1) for net in nets]
    y_pred = np.argmax(np.bincount(y_preds))
    acc += y_true.eq(y_pred).sum()

print(acc / 42000.0)
