import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC, FCPlanenet, FCPlanenet2
from torch_scatter import scatter_mean, scatter_max
from im2mesh.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, positional_encoding, normalize_dynamic_plane_coordinate
from im2mesh.encoder.unet import UNet
from im2mesh.unet3d.model import UNet3D
import pdb


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, **kwargs):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c

class LocalResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1):
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

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

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

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False,
                 unet3d_kwargs=None, plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1,
                 n_blocks=5, pos_encoding=False):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_2 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid,
                                    self.reso_grid)  # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        ##################

        # net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

class DynamicLocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=None,
                 grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, pos_encoding=False, n_channels=3):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        #self.block_2 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.fc_plane_net = FCPlanenet(n_dim=dim, n_channels=n_channels, hidden_dim=hidden_dim)
        self.fc_plane_hdim = nn.Linear(n_channels * 3, hidden_dim)

        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')
        
        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_dynamic_plane_features(self, p, c, basis_normalizer_matrix):
        # acquire indices of features in plane
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def ChangeBasis(self, plane_parameters, device='cuda'):
        # Input: Plane parameters (batch_size x L x 3) - torch.tensor dtype = torch.float32
        # Output: C_mat (batch_size x L x 4 x 3)
        # C_mat is stacked matrices of:
        # 1. Change of basis matrices (batch_size x L x 3 x 3)
        # 2. Normalizing constants (batch_size x L x 1 x 3)
        device = device
        batch_size, L, _ = plane_parameters.size()
        normal = plane_parameters.reshape([batch_size * L, 3]).float()
        normal = normal / torch.norm(normal, p=2, dim=1).view(batch_size * L, 1)  # normalize
        normal = normal + 0.000001  # Avoid non-invertible matrix down the road

        basis_x = torch.tensor([1, 0, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        basis_y = torch.tensor([0, 1, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        basis_z = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)

        v = torch.cross(basis_z.to(device), normal, dim=1)
        zero = torch.zeros([batch_size * L], dtype=torch.float32).to(device)
        skew = torch.zeros([batch_size * L, 3, 3], dtype=torch.float32).to(device)
        skew[range(batch_size * L), 0] = torch.stack([zero, -v[:, 2], v[:, 1]]).t()
        skew[range(batch_size * L), 1] = torch.stack([v[:, 2], zero, -v[:, 0]]).t()
        skew[range(batch_size * L), 2] = torch.stack([-v[:, 1], v[:, 0], zero]).t()

        idty = torch.eye(3).to(device)
        idty = idty.reshape((1, 3, 3))
        idty = idty.repeat(batch_size * L, 1, 1)
        dot = (1 - torch.sum(normal * basis_z, dim=1)).unsqueeze(1).unsqueeze(2)
        div = torch.norm(v, p=2, dim=1) ** 2
        div = div.unsqueeze(1).unsqueeze(2)

        R = (idty + skew + torch.matmul(skew, skew) * dot / div)

        new_basis_x = torch.bmm(R, basis_x.unsqueeze(2))
        new_basis_y = torch.bmm(R, basis_y.unsqueeze(2))
        new_basis_z = torch.bmm(R, basis_z.unsqueeze(2))

        new_basis_matrix = torch.cat([new_basis_x, new_basis_y, new_basis_z], dim=2)

        C_inv = torch.inverse(new_basis_matrix)

        # Define normalization constant
        b_x = torch.abs(new_basis_x).squeeze(2)
        b_y = torch.abs(new_basis_y).squeeze(2)
        p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        p_x = torch.sum(b_x * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_x * b_x, dim=1).unsqueeze(1) * b_x
        p_y = torch.sum(b_y * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_y * b_y, dim=1).unsqueeze(1) * b_y

        c_x = torch.norm(p_x, p=2, dim=1)
        c_y = torch.norm(p_y, p=2, dim=1)

        normalizer = torch.max(c_x, c_y).unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)
        C_mat = torch.cat([C_inv, normalizer], dim=1)
        C_mat = C_mat.view(batch_size, L, 4, 3)

        return C_mat


    def forward(self, p):
        batch_size, T, D = p.size()
        self.device = 'cuda'

        # Plane predictor network
        net_pl = self.fc_plane_net(p)

        plane_parameters = net_pl.view(batch_size, -1, 3)  # plane parameter (batch_size x L x 3)
        C_mat = self.ChangeBasis(plane_parameters, device=self.device) # change of basis and normalizer matrix (concatenated)
        num_planes = C_mat.size()[1]

        net_pl = self.fc_plane_hdim(self.actvn(net_pl))
        net_pl = net_pl.unsqueeze(1)

        # acquire the index for each point
        coord = {}
        index = {}

        for l in range(num_planes):
            coord['plane{}'.format(l)] = normalize_dynamic_plane_coordinate(p.clone(), C_mat[:,l], padding=self.padding)
            index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        ##################
        
        # net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        c = c + net_pl

        fea = {}

        for l in range(C_mat.size()[1]):
            fea['plane{}'.format(l)] = self.generate_dynamic_plane_features(p, c, C_mat[:,l])

        fea['c_mat'] = C_mat

        return fea


class DynamicLocalPoolPointnet2(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks
        for each local point on the ground plane.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False,
                 unet3d_kwargs=None, plane_resolution=None,
                 grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, pos_encoding=False, n_channels=3):
        super().__init__()
        self.c_dim = c_dim
        self.num_channels = n_channels

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        # self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        # self.block_2 = ResnetBlockFC(2*hidden_dim, c_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.fc_plane_net = FCPlanenet2(n_dim=dim, hidden_dim=hidden_dim)

        # Create FC layers based on the number of planes
        self.plane_params = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for i in range(n_channels)
        ])

        self.plane_params_hdim = nn.ModuleList([
            nn.Linear(3, hidden_dim) for i in range(n_channels)
        ])


        self.actvn = nn.ReLU()
        # self.pool = maxpool
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_dynamic_plane_features(self, p, c, basis_normalizer_matrix):
        # acquire indices of features in plane
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix,
                                                padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_dynamic_plane_features2(self, p, c, normal_feature, basis_normalizer_matrix):
        # acquire indices of features in plane
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix,
                                                padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)

        c = c.permute(0, 2, 1)  # B x 512 x T
        c = c + normal_feature.unsqueeze(2)
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid,
                                    self.reso_grid)  # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def ChangeBasis(self, plane_parameters, device='cuda'):
        # Input: Plane parameters (batch_size x L x 3) - torch.tensor dtype = torch.float32
        # Output: C_mat (batch_size x L x 4 x 3)
        # C_mat is stacked matrices of:
        # 1. Change of basis matrices (batch_size x L x 3 x 3)
        # 2. Normalizing constants (batch_size x L x 1 x 3)
        device = device
        batch_size, L, _ = plane_parameters.size()
        normal = plane_parameters.reshape([batch_size * L, 3]).float()
        normal = normal / torch.norm(normal, p=2, dim=1).view(batch_size * L, 1)  # normalize
        normal = normal + 0.000001  # Avoid non-invertible matrix down the road

        basis_x = torch.tensor([1, 0, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        basis_y = torch.tensor([0, 1, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        basis_z = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)

        v = torch.cross(basis_z.to(device), normal, dim=1)
        zero = torch.zeros([batch_size * L], dtype=torch.float32).to(device)
        skew = torch.zeros([batch_size * L, 3, 3], dtype=torch.float32).to(device)
        skew[range(batch_size * L), 0] = torch.stack([zero, -v[:, 2], v[:, 1]]).t()
        skew[range(batch_size * L), 1] = torch.stack([v[:, 2], zero, -v[:, 0]]).t()
        skew[range(batch_size * L), 2] = torch.stack([-v[:, 1], v[:, 0], zero]).t()

        idty = torch.eye(3).to(device)
        idty = idty.reshape((1, 3, 3))
        idty = idty.repeat(batch_size * L, 1, 1)
        dot = (1 - torch.sum(normal * basis_z, dim=1)).unsqueeze(1).unsqueeze(2)
        div = torch.norm(v, p=2, dim=1) ** 2
        div = div.unsqueeze(1).unsqueeze(2)

        R = (idty + skew + torch.matmul(skew, skew) * dot / div)

        new_basis_x = torch.bmm(R, basis_x.unsqueeze(2))
        new_basis_y = torch.bmm(R, basis_y.unsqueeze(2))
        new_basis_z = torch.bmm(R, basis_z.unsqueeze(2))

        new_basis_matrix = torch.cat([new_basis_x, new_basis_y, new_basis_z], dim=2)

        C_inv = torch.inverse(new_basis_matrix)

        # Define normalization constant
        b_x = torch.abs(new_basis_x).squeeze(2)
        b_y = torch.abs(new_basis_y).squeeze(2)
        p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
        p_x = torch.sum(b_x * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_x * b_x, dim=1).unsqueeze(1) * b_x
        p_y = torch.sum(b_y * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_y * b_y, dim=1).unsqueeze(1) * b_y

        c_x = torch.norm(p_x, p=2, dim=1)
        c_y = torch.norm(p_y, p=2, dim=1)

        normalizer = torch.max(c_x, c_y).unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)
        C_mat = torch.cat([C_inv, normalizer], dim=1)
        C_mat = C_mat.view(batch_size, L, 4, 3)

        return C_mat

    def forward(self, p):
        batch_size, T, D = p.size()
        self.device = 'cuda'

        # Plane predictor network
        net_pl = self.fc_plane_net(p)

        normal_fea = []
        normal_fea_hdim = {}

        for l in range(self.num_channels):
            normal_fea.append(self.plane_params[l](self.actvn(net_pl)))
            normal_fea_hdim['plane{}'.format(l)] = self.plane_params_hdim[l](normal_fea[l])

        plane_parameters = torch.stack(normal_fea, dim=1) # plane parameter (batch_size x L x 3)
        C_mat = self.ChangeBasis(plane_parameters,
                                 device=self.device)  # change of basis and normalizer matrix (concatenated)
        num_planes = C_mat.size()[1]

        # acquire the index for each point
        coord = {}
        index = {}

        for l in range(num_planes):
            coord['plane{}'.format(l)] = normalize_dynamic_plane_coordinate(p.clone(), C_mat[:, l],
                                                                            padding=self.padding)
            index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        ##################

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        c = c

        fea = {}

        for l in range(C_mat.size()[1]):
            fea['plane{}'.format(l)] = self.generate_dynamic_plane_features2(p, c, normal_fea_hdim['plane{}'.format(l)], C_mat[:, l])

        fea['c_mat'] = C_mat

        return fea
    