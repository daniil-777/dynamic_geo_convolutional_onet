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

'''
 Set cuda = True. If False, change:
 mat = torch.tensor(mat, device='cuda')
 grid_feature = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')
 counter = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')
 
 to:
 mat = torch.tensor(mat)
 grid_feature = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]])
 counter = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]])
 
 and delete:
 net = net.to(device)
 p_project = p_project.to(device)
'''
is_cuda = True

cfg = config.load_config('configs/pointcloud/onet.yaml', 'configs/default.yaml')
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
#exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Hack because of previous bug in code
# TODO: remove, because shouldn't be necessary
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

# TODO: remove this switch
# metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

# Get sample points
for batch in train_loader:
    a = batch
p = a['points']

# Initialize Encoder functions
import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
import numpy as np

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def SingleChangeBasisMatrix(single_plane_parameter):
    a, b, c, _ = single_plane_parameter
    a, b, c = float(a), float(b), float(c)

    normal = np.array([a, b, c])
    normal = normal / np.sqrt(np.sum(normal ** 2))

    if sum(normal == np.array([0,0,1])) != 3:
        basis_x = np.array([1, 0, 0])
        basis_y = np.array([0, 1, 0])
        basis_z = np.array([0, 0, 1])

        # Construct rotation matrix to align z-axis basis to plane normal
        v = np.cross(basis_z, normal)  # Need to add exception, if normal = [0, 0, 1]. don't do basis rotation
        ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.identity(3) + ssc + np.matmul(ssc, ssc) * (1 - np.dot(normal, basis_z)) / (np.linalg.norm(v) ** 2)

        # Change basis to plane normal basis
        # plane equation in new basis: z = 0
        new_basis_x = np.array([np.matmul(R, basis_x)])  # plane normal basis in standard coordinate
        new_basis_y = np.array([np.matmul(R, basis_y)])
        new_basis_z = np.array([np.matmul(R, basis_z)])
        new_basis_matrix = np.concatenate((new_basis_x.T, new_basis_y.T, new_basis_z.T), axis=1)

        C_inv = np.linalg.inv(new_basis_matrix)

    else:
        C_inv = np.identity(3)

    return C_inv

def ChangeBasisMatrix(plane_parameters):
    # Input: Plane parameters (Lx4)
    # Output: Change of basis matrices (L x 3 x 3)
    L = len(plane_parameters)
    mat = SingleChangeBasisMatrix(plane_parameters[0])

    for i in range(1, L):
        mat = np.vstack((mat, SingleChangeBasisMatrix(plane_parameters[i])))

    mat = mat.reshape((L,3,3)).T
    mat = torch.tensor(mat, device='cuda')
    return mat

c_dim = 128
dim = 3
hidden_dim = 128
fc_pos = nn.Linear(dim, 2*hidden_dim)
block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
fc_c = nn.Linear(hidden_dim, c_dim)

actvn = nn.ReLU()
pool = maxpool

# Feed forward

## Output:
### grid_feature: size batch_size x L x H x W x d
### L = number of channels, H = Grid heights, W = Grid Weights, d = number of features per grid
### C_mat: size L x 3 x 3 (to be used for query point projection)

net = fc_pos(p)
net = block_0(net)
pooled = pool(net, dim=1, keepdim=True).expand(net.size())
net = torch.cat([net, pooled], dim=2)

net = block_1(net)
pooled = pool(net, dim=1, keepdim=True).expand(net.size())
net = torch.cat([net, pooled], dim=2)

net = block_2(net)
pooled = pool(net, dim=1, keepdim=True).expand(net.size())
net = torch.cat([net, pooled], dim=2)

net = block_3(net)
pooled = pool(net, dim=1, keepdim=True).expand(net.size())
net = torch.cat([net, pooled], dim=2)

net = block_4(net)

# Recude to  B x F
#net = self.pool(net, dim=1)

#c = self.fc_c(self.actvn(net))

# Assume have plane parameters of size Lx4
plane_param_np = np.array([[0.5, 0.5, 0.5, 0.2], [1, 0, 0, 2], [1, 4, 2, 5]])
plane_parameters = torch.tensor(plane_param_np)

C_mat = ChangeBasisMatrix(plane_parameters)  # L x 3 x 3

# Create grid feature
grid_res = 64 # If CUDA error, reduce to lower number
max_dim = 0.55
H = grid_res + 1
W = grid_res + 1
interval = float(2 / (grid_res))

grid_feature = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')
counter = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')

net = net.to(device)
for l in range(C_mat.size()[0]):
    p_project = torch.div(p, max_dim)
    p_project = p_project.to(device)
    p_project = torch.matmul(p_project.double(), C_mat[l])
    p_project = p_project[:, :, 0:2] / np.sqrt(3)  # divide by sqrt(3) so that range is [-1,1]
    xy_index = torch.round((p_project + 1) / interval).int()

    for n in range(p.size()[1]):
        x_grid, y_grid = xy_index[:, n, 0], xy_index[:, n, 1]
        x_grid = x_grid.tolist()
        y_grid = y_grid.tolist()
        counter[range(p.size(0)), l, x_grid, y_grid] = counter[range(p.size(0)), l, x_grid, y_grid] + 1
        grid_feature[range(p.size(0)), l, x_grid, y_grid] = grid_feature[range(p.size(0)), l, x_grid, y_grid] + \
                                                            net[range(p.size(0)), n]

# Average overlapping projection
counter[counter == 0] = 1
grid_feature = torch.div(grid_feature, counter)

# U-net
# Output: torch.Size([16, 3, 32, 32, 128])
##

# Bilinear interpolation, given point p, where p is real number
# p is x, y, z
# change the basis to plane 1, you get p_new1

#I assume that this function take a projected point on the grid
#probably it's just for one feature (last dimension for the grid)


def bilinear_interpolation(x, y, grid_size, w, h, grid, batch, plane):
    """calculates bilinear interpolated feature using neighbouring features on verteces
       Parameters
        ----------
        x         : float
                          x coordinate of the projected point
        y         : float
                          y coordinate of the projected point
        grid_size : float 
                          size of grid's edge
        w         : float 
                          width of the grid
        h         : float 
                          height of the grid
        grid.     : torch.tensor
                          [batch, plane, width, height, features] - tensor of features
        batch     : int
                          index of a batch
        plane.    : int
                          index of a plane
        ---------
        Returns
        feature   : float
    """
    a = int(x / grid_size)
    b = int(y / grid_size)
   
    # a if not in the the rightest position, a - 1- in another case
    x_left = a - 1 + int(a < w)
    x_right = a + int(a < w)
    # b if not in the the highest position, b - 1 - in another case
    y_low = b - 1 + int(b < w)
    y_high = b + int(b < w)
    feature_11 = grid[batch, plane, x_left, y_low, :]
    feature_12 = grid[batch, plane, x_left, y_high, :]
    feature_21 = grid[batch, plane, x_right, y_low, :]
    feature_22 = grid[batch, plane, x_right, y_high, :]
    inter_feature = (feature_11 * (x_right - x) * (y_high - y) +
                     feature_21 * (x - x_left) * (y_high - y) +
                     feature_12 * (x_right - x) * (y - y_low) +
                     feature_22 * (x - x_left) * (y - y_low)
                     ) / (x_right - x_left)*(y_high - y_low)

    return inter_feature

