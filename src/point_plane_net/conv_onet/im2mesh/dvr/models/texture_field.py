
import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, ResnetBlockConv1d
)


class TextureField(nn.Module):
    ''' Texture Field decoder class.

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
        self.fc_out = nn.Linear(hidden_size, 3)

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

    def forward(self, p, z=None, c=None, **kwargs):
        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.z_dim != 0 and z is not None:
                net_z = self.fc_z[n](z).unsqueeze(1)
                net = net + net_z

            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c).unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))
        # out = sigmoid(out)
        return out


class TextureFieldCBatchNorm(nn.Module):
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
        self.conv_out = nn.Conv1d(hidden_size, 3, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z=None, c=None, **kwargs):

        p = p.transpose(-2, -1)
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = sigmoid(out).transpose(-2, -1)
        return out


class TextureFieldBatchNorm(nn.Module):
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
        self.fc_out = nn.Linear(hidden_size, 3)
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

    def forward(self, p, z=None, c=None, **kwargs):
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
        out = sigmoid(out)
        return out