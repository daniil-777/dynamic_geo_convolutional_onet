import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)
from im2mesh.common import normalize_coordinate, normalize_3d_coordinate, coordinate2index, positional_encoding, normalize_dynamic_plane_coordinate
import pdb


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
                 hidden_size=128, leaky=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()

        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False, padding=0.1):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class LocalDecoderCBatchNorm(nn.Module):
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
                 hidden_size=256, leaky=False, legacy=False, bilinear=True):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.bilinear = bilinear # use bilinear interplolation on the plane features

    def indexing_plane_feature(self, p, c, plane='xz'):
        reso = c.size(-1)
        if self.bilinear:
            xy = reso * normalize_coordinate(p.clone(), plane=plane) # normalize to the range of (0, 1)
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

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class DecoderCBatchNorm2(nn.Module):
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

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


class DecoderCBatchNormNoResnet(nn.Module):
    ''' Decoder CBN with no ResNet blocks class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_0 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_3 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_4 = nn.Conv1d(hidden_size, hidden_size, 1)

        self.bn_0 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_1 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_2 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_3 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_4 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_5 = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.actvn(self.bn_0(net, c))
        net = self.fc_0(net)
        net = self.actvn(self.bn_1(net, c))
        net = self.fc_1(net)
        net = self.actvn(self.bn_2(net, c))
        net = self.fc_2(net)
        net = self.actvn(self.bn_3(net, c))
        net = self.fc_3(net)
        net = self.actvn(self.bn_4(net, c))
        net = self.fc_4(net)
        net = self.actvn(self.bn_5(net, c))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out


class DecoderBatchNorm(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = ResnetBlockConv1d(hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)
        self.block3 = ResnetBlockConv1d(hidden_size)
        self.block4 = ResnetBlockConv1d(hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out


class LocalPointDecoder(nn.Module):
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
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

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

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val']**2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        #p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)**2
            weight = (dist/self.var).exp() # Guassian kernel
        else:
            weight = 1/((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)

        #weight normalization
        weight = weight/weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea # B x M x c_dim

        return c_out

    def forward(self, p, z, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            #pdb.set_trace()
            c = torch.cat(c_list, dim=1)

        else:
           if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

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


class LocalDecoder2(nn.Module):
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
                 hidden_size=32, leaky=False, sample_mode='bilinear', n_blocks=3, padding=0.1, feat_dim=32):
        ''' I added a feat_dim attribute
        '''
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(feat_dim, hidden_size) for i in range(n_blocks)
            ])
            self.conv_layers = nn.ModuleList([
                nn.ConvTranspose2d(z_dim, 256, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1,),
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

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def upsample_latent_code_to_feature_map(self, z_plane):
        batch_size = z_plane.shape[0]
        n_planes = 3
        z = z_plane.reshape(batch_size, -1, 1, 1)
        net = self.conv_layers[0](z)
        for conv_layer in self.conv_layers[1:]:
            net = conv_layer(self.actvn(net))

        out_dict = {
            'xz': net#[:, :32],
            #'xy': net[:, 32:64],
            #'yz': net[:, 64:],
        }
        return out_dict




    def forward(self, p, z_plane, c_plane, **kwargs):

        if self.z_dim > 0:
            # I do the reshaping
            z_plane = self.upsample_latent_code_to_feature_map(z_plane)
            plane_type = list(z_plane.keys())
            if 'grid' in plane_type:
                z = self.sample_grid_feature(p, z_plane['grid'])
            if 'xz' in plane_type:
                z = self.sample_plane_feature(p, z_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                z += self.sample_plane_feature(p, z_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                z += self.sample_plane_feature(p, z_plane['yz'], plane='yz')
            z = z.transpose(1, 2)

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if 'grid' in plane_type:
                c = self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c = self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
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
                 hidden_size=256, leaky=False, sample_mode='bilinear', n_blocks=5, pos_encoding=False, padding=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])
            self.conv_layers = nn.ModuleList([
                nn.ConvTranspose2d(2, 6, 3, stride=2, padding=1, output_padding=1, ),
                nn.ConvTranspose2d(6, 12, 3, stride=2, padding=1, output_padding=1, ),
                nn.ConvTranspose2d(12, 24, 3, stride=2, padding=1, output_padding=1, ),
                nn.ConvTranspose2d(24, 48, 3, stride=2, padding=1, output_padding=1, ),
                nn.ConvTranspose2d(48, 96, 3, stride=2, padding=1, output_padding=1, ),
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

        self.sample_mode = sample_mode
        self.padding = padding

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
            -1).squeeze(-1)
        return c

    def upsample_latent_code_to_feature_map(self, z_plane):
        batch_size = z_plane.shape[0]
        n_planes = 3

        z = z_plane.reshape(batch_size, 2, 4, 4)
        net = self.conv_layers[0](z)
        for conv_layer in self.conv_layers[1:]:
            net = conv_layer(self.actvn(net))

        out_dict = {
            'xz': net[:, :32],
            'xy': net[:, 32:64],
            'yz': net[:, 64:],
        }
        return out_dict

    def forward(self, p, z_plane, c_plane, **kwargs):

        if self.z_dim > 0:
            # I do the reshaping
            z_plane = self.upsample_latent_code_to_feature_map(z_plane)
            plane_type = list(z_plane.keys())
            if 'grid' in plane_type:
                z = self.sample_grid_feature(p, z_plane['grid'])
            if 'xz' in plane_type:
                z = self.sample_plane_feature(p, z_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                z += self.sample_plane_feature(p, z_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                z += self.sample_plane_feature(p, z_plane['yz'], plane='yz')
            z = z.transpose(1, 2)

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        p = p.float()
        ##################
        if self.pos_encoding:
            p = self.pe(p)
        ##################

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


class DynamicLocalDecoder(nn.Module):
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
                 hidden_size=256, leaky=False, sample_mode='bilinear', n_blocks=5, pos_encoding=False, padding=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])
            self.conv_layers = nn.ModuleList([
                nn.ConvTranspose2d(2, 6, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(6, 12, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(12, 24, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(24, 48, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(48, 96, 3, stride=2, padding=1, output_padding=1,),
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

        self.sample_mode = sample_mode
        self.padding = padding

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_dynamic_plane_feature(self, p, c, basis_normalizer_matrix):
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def upsample_latent_code_to_feature_map(self, z_plane):
        batch_size = z_plane.shape[0]
        n_planes = 3

        z = z_plane.reshape(batch_size, 2, 4, 4)
        net = self.conv_layers[0](z)
        for conv_layer in self.conv_layers[1:]:
            net = conv_layer(self.actvn(net))

        out_dict = {
            'xz': net[:, :32],
            'xy': net[:, 32:64],
            'yz': net[:, 64:],
        }
        return out_dict


    def forward(self, p, z_plane, c_plane, **kwargs):

        if self.z_dim > 0:
            # I do the reshaping
            z_plane = self.upsample_latent_code_to_feature_map(z_plane)

            num_planes = z_plane['c_mat'][1]

            z = 0
            for l in range(num_planes):
                z += self.sample_dynamic_plane_feature(p, z_plane['plane{}'.format(l)], z_plane['c_mat'][:,l])

            z = z.transpose(1, 2)

        if self.c_dim != 0:
            c = 0
            num_planes = c_plane['c_mat'].size()[1]

            for l in range(num_planes):
                c += self.sample_dynamic_plane_feature(p, c_plane['plane{}'.format(l)], c_plane['c_mat'][:,l])

            c = c.transpose(1, 2)

        p = p.float()
        ##################
        if self.pos_encoding:
            p = self.pe(p)
        ##################
        
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


class LocalConvDecoder(nn.Module):
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
                 hidden_size=32, leaky=False, sample_mode='bilinear', n_blocks=3, padding=0.1, feat_dim=32,
                 n_conv_layer=4, plane_type=['xz']):
        ''' I added a feat_dim attribute
        '''
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.plane_type = plane_type

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(feat_dim, hidden_size) for i in range(n_blocks)
            ])


        if z_dim != 0:
            res = self.z_dim * (2 ** n_conv_layer)
            self.conv_layers_z = []
            for i in range(n_conv_layer):
                if 'grid' in plane_type:
                    self.conv_layers_z.append(
                        nn.ConvTranspose3d(res, int(res / 2), 2, 2)
                    )
                    res = int (res / 2)
                else:
                    self.conv_layers_z.append(
                        nn.ConvTranspose2d(res, int(res / 2), 2, 2)
                    )
                    res = int (res / 2)
            self.conv_layers_z = nn.ModuleList(self.conv_layers_z)


        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


            # self.conv_layers_c = []
            # res = self.c_dim * ( 2 ** n_conv_layer)
            # for i in range(n_conv_layer):
            #     # TODO FIX
            #     if False: #'grid' in plane_type:
            #         self.conv_layers_c.append(
            #             nn.ConvTranspose3d(res, int(res / 2), 2, 2)
            #         )
            #         res = int (res / 2)
            #     else:
            #         self.conv_layers_c.append(
            #             nn.ConvTranspose2d(res, int(res / 2), 2, 2)
            #         )
            #         res = int (res / 2)
            # self.conv_layers_c = nn.ModuleList(self.conv_layers_c)

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        #c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        c = F.grid_sample(c, vgrid, padding_mode='border', mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        #c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        c = F.grid_sample(c, vgrid, padding_mode='border', mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, z_plane, c_plane, **kwargs):
        add_dim = False
        if len(p.shape) == 2:
            p = p.unsqueeze(0)
            add_dim = True
        if self.z_dim > 0:
            if not (type(z_plane) == dict):
                if 'grid' in self.plane_type:
                    z_plane = {'grid': z_plane}
                else:
                    z_plane = {'xz': z_plane}
            for (k, v) in z_plane.items():
                net = self.conv_layers_z[0](v)
                for conv_layer in self.conv_layers_z[1:]:
                    net = conv_layer(self.actvn(net))
                z_plane[k] = net

            plane_type = list(z_plane.keys())
            if 'grid' in plane_type:
                z = self.sample_grid_feature(p, z_plane['grid'])
            if 'xz' in plane_type:
                z = self.sample_plane_feature(p, z_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                z += self.sample_plane_feature(p, z_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                z += self.sample_plane_feature(p, z_plane['yz'], plane='yz')
            z = z.transpose(1, 2)

        if self.c_dim != 0:
            #c_plane_out = {}
            if not (type(c_plane) == dict):
                # TODO
                if False: #'grid' in self.plane_type:
                    c_plane = {'grid': c_plane}
                else:
                    c_plane = {'xz': c_plane}
            # for (k, v) in c_plane.items():
            #     net = self.conv_layers_c[0](v)
            #     for conv_layer in self.conv_layers_c[1:]:
            #         net = conv_layer(self.actvn(net))
            #     c_plane[k] = net


            plane_type = list(c_plane.keys())
            if 'grid' in plane_type:
                c = self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c = self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
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

        if add_dim:
            out = out.squeeze(0)
        return out
