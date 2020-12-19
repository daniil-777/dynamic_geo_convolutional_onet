## 3D Vision course implementation, most of it is being reused from the train.py file
## Author: Dusan

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import torch.nn as nn
## TODO : to be done when succesfully connected all needed parts
from im2mesh.encoder import point_plane_net
new_model = point_plane_net.PointPlaneResnet(k = 5).cuda()
# new_model.cuda()
input = torch.randn(10,20, 3).cuda()
labels = torch.randn(10,128).cuda() * 1000
input.size()
outputs = new_model(input)
print(outputs.shape)