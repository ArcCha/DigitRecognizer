from pathlib import Path

from net import *

nets = [CNN() for _ in range(35)]
net_dirs = [Path('./' + type(net).__name__ + str(i))
            for i, net in enumerate(nets)]
