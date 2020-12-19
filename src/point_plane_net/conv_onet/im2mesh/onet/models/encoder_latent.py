import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC
from torch_scatter import scatter_mean
from im2mesh.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate


# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        self.fc_0 = nn.Linear(1, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, p, x, c=None, **kwargs):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_0(x.unsqueeze(-1))
        net = net + self.fc_pos(p)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce
        #  to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class EncoderPointNet(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, p, x, c=None, inputs=None, **kwargs):
        batch_size, T, D = inputs.size()

        # output size: B x T X F
        net = self.fc_pos(inputs)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce
        #  to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class ConvEncoder(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, z_dim=128,  dim=3, hidden_dim=128, plane_resolution=512, plane_type=['xz'], padding=0.1,
                 n_conv_layer=4, **kwargs):
        super().__init__()
        self.z_dim = z_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, z_dim)
        self.fc_c = nn.Linear(hidden_dim, z_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        if 'grid' in plane_type:
            self.conv_layers = []
            res = self.z_dim
            for i in range(n_conv_layer):
                self.conv_layers.append(
                    nn.Conv3d(res, res * 2, 3, 2, 1)
                )
                res *= 2
            self.conv_layers = nn.ModuleList(self.conv_layers)
            self.mean = nn.Conv3d(res, res, 1, 1, 0)
            self.log_std = nn.Conv3d(res, res, 1, 1, 0)
        else:
            print("Currently only single plane is supported!")
            self.conv_layers = []
            res = self.z_dim
            for i in range(n_conv_layer):
                self.conv_layers.append(
                    nn.Conv2d(res, res * 2, 3, 2, 1)
                )
                res *= 2
            self.conv_layers = nn.ModuleList(self.conv_layers)
            self.mean = nn.Conv2d(res, res, 1, 1, 0)
            self.log_std = nn.Conv2d(res, res, 1, 1, 0)

        self.reso = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.z_dim, self.reso**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.z_dim, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        # if self.unet is not None:
        # fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.z_dim, self.reso**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.z_dim, self.reso, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # if self.unet3d is not None:
        # fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, p_occ, occ, c=None, inputs=None, **kwargs):
        p = inputs
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p) # B x T x 1024
        net = self.block_0(net) # B x T x 512
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) # B x T x 512
        net = torch.cat([net, pooled], dim=2) # B x T x 1024

        net = self.block_1(net) # B x T x 512
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) # B x T x 512
        net = torch.cat([net, pooled], dim=2) # B x T x 1024

        net = self.block_2(net) # B x T x 512
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) # B x T x 512
        net = torch.cat([net, pooled], dim=2) # B x T x 1024

        net = self.block_3(net) # B x T x 512
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) # B x T x 512
        net = torch.cat([net, pooled], dim=2) # B x T x 1024

        c = self.block_4(net) # B x T x 512

        if 'grid' in self.plane_type:
            # fea_grid = {}
            # fea_grid['grid'] = 
            net = self.generate_grid_features(p, c)
        else:
            fea_plane = {}
            if 'xz' in self.plane_type:
                fea_plane['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea_plane['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea_plane['yz'] = self.generate_plane_features(p, c, plane='yz')
            # TODO currently only working for single plane / grid     
            net = fea_plane['xz']

        # Apply convolutional layers
        net = self.conv_layers[0](net)
        for conv_layer in self.conv_layers[1:]:
            net = conv_layer(self.actvn(net))

        mean = self.mean(net)
        log_std = self.log_std(net)

        return mean, log_std

class ConvEncoder2(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, z_dim=128,  dim=3, hidden_dim=128, plane_resolution=512, plane_type=['xz'], padding=0.1,
                 n_conv_layer=4, n_blocks=3, **kwargs):
        super().__init__()
        self.z_dim = z_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_z = nn.Linear(2*hidden_dim, z_dim)
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_4 = ResnetBlockFC(2*hidden_dim, z_dim)
        # self.fc_c = nn.Linear(hidden_dim, z_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        if 'grid' in plane_type:
            self.conv_layers = []
            res = self.z_dim
            for i in range(n_conv_layer):
                self.conv_layers.append(
                    nn.Conv3d(res, res * 2, 3, 2, 1)
                )
                res *= 2
            self.conv_layers = nn.ModuleList(self.conv_layers)
            self.mean = nn.Conv3d(res, res, 1, 1, 0)
            self.log_std = nn.Conv3d(res, res, 1, 1, 0)
        else:
            print("Currently only single plane is supported!")
            self.conv_layers = []
            res = self.z_dim
            for i in range(n_conv_layer):
                self.conv_layers.append(
                    nn.Conv2d(res, res * 2, 3, 2, 1)
                )
                res *= 2
            self.conv_layers = nn.ModuleList(self.conv_layers)
            self.mean = nn.Conv2d(res, res, 1, 1, 0)
            self.log_std = nn.Conv2d(res, res, 1, 1, 0)

        self.reso = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.z_dim, self.reso**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.z_dim, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        # if self.unet is not None:
        # fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.z_dim, self.reso**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.z_dim, self.reso, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # if self.unet3d is not None:
        # fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, p_occ, occ, c=None, inputs=None, **kwargs):
        p = inputs
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p) # B x T x 1024

        for block in self.blocks:
            net = block(net)
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size()) # B x T x 512
            net = torch.cat([net, pooled], dim=2) # B x T x 1024         

        c = self.fc_z(net) # B x T x 512

        if 'grid' in self.plane_type:
            # fea_grid = {}
            # fea_grid['grid'] = 
            net = self.generate_grid_features(p, c)
        else:
            fea_plane = {}
            if 'xz' in self.plane_type:
                fea_plane['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea_plane['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea_plane['yz'] = self.generate_plane_features(p, c, plane='yz')
            # TODO currently only working for single plane / grid     
            net = fea_plane['xz']

        # Apply convolutional layers
        net = self.conv_layers[0](net)
        for conv_layer in self.conv_layers[1:]:
            net = conv_layer(self.actvn(net))

        mean = self.mean(net)
        log_std = self.log_std(net)
        return mean, log_std
