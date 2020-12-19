import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)


class Decoder_interpolation(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=32,
                 hidden_size=256):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.c_dim = c_dim

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = ResnetBlockFC(hidden_size, c_dim)
        self.block1 = ResnetBlockFC(c_dim, c_dim)
        self.block2 = ResnetBlockFC(c_dim, c_dim)
        self.block3 = ResnetBlockFC(c_dim, c_dim)
        self.block4 = ResnetBlockFC(c_dim, c_dim)

        self.fc_out = nn.Conv1d(c_dim, 1, 1)
        self.actvn = F.relu

    def forward(self, p, z, c, C_mat, **kwargs):

        batch_size, T, d = p.size()
        _, L, H, W, d_dim = c.size()

        # Billinear Interpolation
        max_dim = 0.55

        interval = float(2 / (H-1))

        c = self.BillinearInterpolation(p, c, C_mat, max_dim, interval)

        p = p.transpose(1, 2)
        net = self.fc_p(p)
        net = net.transpose(1, 2)

        net = self.block0(net)
        net = net + c
        net = self.block1(net)
        net = net + c
        net = self.block2(net)
        net = net + c
        net = self.block3(net)
        net = net + c
        net = self.block4(net)
        net = net + c

        out = self.fc_out(self.actvn(net.transpose(1, 2)))
        #print(self.fc_out.weight)
        out = out.squeeze(1)
        return out

        
    def BillinearInterpolation(self, p, net, C_mat, max_dim, interval, device='cuda'):
        # p: points (batch_size, num_of_points, dimension)
        # c: U-net feature
        # C_mat: C inverse matrices
        # max_dim = maximum range of point
        # interval = step size in the grid
        # device = 'cuda'
        # p = p.to(device)

        #C_mat = C_mat.to(device)
        batch_size, T, d = p.size()
        _, L, H, W, d_dim = net.size()

        interpolated_feature = torch.zeros(
            [batch_size, L, p.size(1), d_dim], device=device)
        #interpolated_feature = torch.zeros([batch_size, L, p.size(1), h_dim]).to('cpu')

        for l in range(C_mat.size()[1]):
            p_project = torch.div(p, max_dim)
            p_project = torch.transpose(p_project, 2, 1)
            p_project = torch.bmm(C_mat[:, l, :3], p_project)
            p_project = torch.transpose(p_project, 2, 1)

            #p_project = p_project / 1.73205
            # divide by normalizer so that range is [-1,1]
            p_project = p_project / \
                (C_mat[:, l, 3, 0]+0.05).unsqueeze(1).unsqueeze(2)
            p_project = p_project[:, :, :2]
            xy_index = (p_project + 1) / interval
            #print((C_mat[:,l,3,0]+0.01).unsqueeze(1).unsqueeze(2))
            #print("DECODER INDEX: ", torch.max(torch.abs(xy_index)))

            for n in range(p.size()[1]):
                x_grid, y_grid = xy_index[:, n, 0], xy_index[:, n, 1]
                x_grid = x_grid.tolist()
                y_grid = y_grid.tolist()

                x_left = np.floor(x_grid).astype(int).tolist()
                x_right = np.ceil(x_grid).astype(int).tolist()
                y_low = np.floor(y_grid).astype(int).tolist()
                y_high = np.ceil(y_grid).astype(int).tolist()

                diff_x = np.array(x_right) - np.array(x_grid)
                diff_x_tensor = torch.tensor(diff_x)[:, None].float().to(device)
                diff_y = np.array(y_high) - np.array(y_grid)
                diff_y_tensor = torch.tensor(diff_y)[:, None].float().to(device)

                feature11 = net[range(batch_size), l, x_left, y_low].to(device)
                feature12 = net[range(batch_size), l, x_right, y_low].to(device)
                feature21 = net[range(batch_size), l, x_left, y_high].to(device)
                feature22 = net[range(batch_size), l, x_right, y_high].to(device)

                inter_feature = (feature11 * diff_x_tensor * diff_y_tensor +
                                feature12 * (1 - diff_x_tensor) * diff_y_tensor +
                                feature21 * diff_x_tensor * (1 - diff_y_tensor) +
                                feature22 * (1 - diff_x_tensor) * (1 - diff_y_tensor))

                interpolated_feature[range(batch_size), l, n] = inter_feature

        interpolated_feature = torch.sum(interpolated_feature, dim=1)

        return interpolated_feature


        
