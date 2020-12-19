
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, ResnetBlockConv1d,
)
# from im2mesh.common import normalize_coordinate, coordinate2index, grid_sample_on_img


class Decoder(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, n_blocks=5, out_dim=1, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.cold = None

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z=None, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.z_dim != 0 and z is not None:
                net_z = self.fc_z[n](z)
                if batchwise:
                    net_z = net_z.unsqueeze(1)
                net = net + net_z

            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        return out


class DecoderBatchNorm(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, n_blocks=5):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.cold = None

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z=None, c=None, only_occupancy=False,
                only_texture=False, **kwargs):
        if len(p.shape) != 3:
            print(p.shape)
        assert(len(p.shape) == 3)
        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.z_dim != 0 and z is not None:
                net_z = self.fc_z[n](z).unsqueeze(1)
                net = net + net_z

            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c).unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net.transpose(-2, -1))
            net = net.transpose(-2, -1)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        if only_occupancy:
            out = out[:, :, 0]
        if only_texture:
            out = out[:, :, 1:]
        return out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=128,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.z_dim = z_dim
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, c_dim)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z=None, c=None, **kwargs):

        p = p.transpose(-2, -1)
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)
        return out

class LocalDecoder(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
        Instead of conditioning on global features, on plane local features

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, bilinear=True, n_blocks=5, padding=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.bilinear = bilinear # use bilinear interplolation on the plane features
        self.padding = padding

    def indexing_plane_feature(self, p, c, plane='xz'):
        reso = c.size(-1)
        if self.bilinear:
            xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
            xy = xy.permute(0, 2, 1)
            xy = xy[..., None]
            c = grid_sample_on_img(c, xy).squeeze(-1) # bilinear interpolation
        else:
            c = c.reshape(1, self.c_dim, reso**2)
            xy = normalize_coordinate(p.clone(), plane=plane)
            ind = coordinate2index(xy, reso) # indexed on ground plane
            c = c.gather(dim=2, index=ind.expand(-1, self.c_dim, -1))
        return c

    def forward(self, p, z, c_plane, **kwargs):
        plane_type = list(c_plane.keys())
        if 'xz' in plane_type:
            c = self.indexing_plane_feature(p, c_plane['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.indexing_plane_feature(p, c_plane['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.indexing_plane_feature(p, c_plane['yz'], plane='yz')

        c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.z_dim != 0:
                net = net + self.fc_z[i](z)
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            
            net = self.blocks[i](net)
    
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out