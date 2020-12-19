import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
from torch_scatter import scatter_mean
from im2mesh.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate
from im2mesh.encoder.unet import UNet
from im2mesh.unet3d.model import UNet3D
import pdb


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class LocalResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, plane_resolution=128, plane_type=['xz'], padding=0.1,
                 n_conv_layer=4, **kwargs):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        if 'grid' in plane_type:
            print('not yet implemented')
            exit(0)
        else:
            self.conv_layers = []
            res = self.c_dim
            for i in range(n_conv_layer):
                self.conv_layers.append(
                    nn.Conv2d(res, res * 2, 3, 2, 1)
                )
                res *= 2
            self.conv_layers = nn.ModuleList(self.conv_layers)

        self.reso = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        # if self.unet is not None:
        # fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso, self.reso, self.reso) # sparce matrix (B x 512 x reso x reso)

        # if self.unet3d is not None:
        # fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, p):
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
            fea_grid = {}
            fea_grid['grid'] = self.generate_grid_features(p, c)
        else:
            fea_plane = {}
            if 'xz' in self.plane_type:
                fea_plane['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea_plane['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea_plane['yz'] = self.generate_plane_features(p, c, plane='yz')

        # Apply convolutional layers
        for (k, v) in fea_plane.items():
            net = self.conv_layers[0](v)
            for conv_layer in self.conv_layers[1:]:
                net = conv_layer(self.actvn(net))
            fea_plane[k] = net
        return fea_plane
