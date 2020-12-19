import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)

def BillinearInterpolationSphere(meridian, latitude, p, net, max_dim, device='cuda'):
    # p: points (batch_size, num_of_points, dimension)
    # c: U-net feature
    # C_mat: C inverse matrices
    # max_dim = maximum range of point
    # interval = step size in the grid
    # device = 'cuda'
    # device = "cpu"
    # p = p.to(device)

    #C_mat = C_mat.to(device)
    batch_size, T, d = p.size()
    _, L, H, W, d_dim = net.size()
    # print("d_dim", d_dim)
    # print("batch_size", batch_size)
    interpolated_feature = torch.zeros([batch_size, L,  p.size(1), d_dim], device=device)
    
    for l in range(1):
        H = latitude
        W = meridian
        # W = 70 #number_meridians
        # H = 70 #number_latitudes
        pi = 3.1415927410125732
        dim_merid_interval = 360 / (W)
        dim_latit_interval = 180 / (H)
        # print("welcome to decoder!")
        x_v = p[:,:,0]
        y_v = p[:,:,1]
        z_v = p[:,:,2]
        latitudes = (90 - torch.atan2(z_v, torch.sqrt(x_v**2 + y_v**2))* 180/pi) #from north to the south from 0 to 180
        meridians = (360 + torch.atan2(y_v, x_v)* 180/pi) % 360 #counter-clockwise from 0 to 360

        y_grid = torch.floor(latitudes / dim_latit_interval)
        x_grid = torch.floor(meridians / dim_merid_interval)

        # x_grid = torch.round(latitudes / dim_latit_interval) #latit_coord
        # y_grid = torch.round(meridians / dim_merid_interval) #merid_coord

        # x_grid[x_grid>=(H-1)] = H-1-0.1
        # x_grid[x_grid<0] = 0
        
        # y_grid[y_grid>=(H-1)] = H-1-0.1
        # y_grid[y_grid<0] = 0

        # print("xgrid shape", x_grid.shape)
        # x_left = torch.roll(x_grid, -1, 1)
        # x_right = torch.roll(x_grid, 1, 1)

        # y_low = torch.roll(y_grid, -1, 1)
        # y_high = torch.roll(y_grid, 1, 1)
 
        x_left = (x_grid - 1) % W
        x_right = (x_grid + 1) % W
        
        y_low = (y_grid - 1) - (y_grid - 1)// H
        y_high = (y_grid + 1) - (y_grid + 1)// H
        
#         x_left = x_grid - 1
#         x_right = x_grid + 1
#         y_low = y_grid - 1
#         y_high = y_grid + 1
        
#         x_left[x_left<0] = W - 1
#         x_right[x_right > W - 1] = 0
        
#         y_low[y_low < 0] = 0
#         y_high[y_high > H - 1] = H - 1
      

        grid_l = net[:,l,:,:]
        # print("x_right",x_left.int().tolist())
        # print("y_low", y_low.int().tolist())
        feature11 = grid_l[:, x_left.int().tolist(), y_low.int().tolist()].to(device)
        feature12 = grid_l[:, x_right.int().tolist(), y_low.int().tolist()].to(device)
        feature21 = grid_l[:, x_left.int().tolist(), y_high.int().tolist()].to(device)
        feature22 = grid_l[:, x_right.int().tolist(), y_high.int().tolist()].to(device)

        # Construct interpolated features
        feature11 = feature11[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature12 = feature12[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature21 = feature21[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature22 = feature22[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        diff_x = torch.flatten(x_right - x_grid).unsqueeze(1)
        diff_y = torch.flatten(y_high - y_grid).unsqueeze(1)

        inter_feature = (feature11 * diff_x * diff_y +
                         feature12 * (1 - diff_x) * diff_y +
                         feature21 * diff_x * (1 - diff_y) +
                         feature22 * (1 - diff_x) * (1 - diff_y))

        inter_feature = inter_feature.view(batch_size, T, d_dim)
        interpolated_feature[:,l,:,:] = inter_feature

    interpolated_feature = torch.sum(interpolated_feature, dim=1)
    # print("shape interp feature", interpolated_feature.shape)
    return interpolated_feature


def BillinearInterpolation(p, net, C_mat, max_dim, interval, device='cuda'):
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

    interpolated_feature = torch.zeros([batch_size, L, p.size(1), d_dim], device=device)

    for l in range(C_mat.size()[1]):
        p_project = torch.div(p, max_dim)
        p_project = torch.transpose(p_project, 2,1)
        p_project = torch.bmm(C_mat[:,l,:3], p_project)
        p_project = torch.transpose(p_project, 2,1)

        p_project = p_project / (C_mat[:,l,3,0]+0.05).unsqueeze(1).unsqueeze(2) # divide by normalizer so that range is [-1,1]
        p_project = p_project[:,:,:2]
        xy_index = (p_project + 1) / interval
        #print(xy_index)
        xy_index[xy_index>=(H-1)] = H-1-0.1
        xy_index[xy_index<0] = 0

        x_grid, y_grid = xy_index[:, :, 0], xy_index[:, :, 1]
        x_left = torch.round(x_grid - 0.5)
        x_right = torch.round(x_grid + 0.5)
        y_low = torch.round(y_grid - 0.5)
        y_high = torch.round(y_grid + 0.5)

        grid_l = net[:,l,:,:]
        feature11 = grid_l[:, x_left.int().tolist(), y_low.int().tolist()].to(device)
        feature12 = grid_l[:, x_right.int().tolist(), y_low.int().tolist()].to(device)
        feature21 = grid_l[:, x_left.int().tolist(), y_high.int().tolist()].to(device)
        feature22 = grid_l[:, x_right.int().tolist(), y_high.int().tolist()].to(device)

        # Construct interpolated features
        feature11 = feature11[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature12 = feature12[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature21 = feature21[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        feature22 = feature22[range(batch_size), range(batch_size)].view(batch_size * T, d_dim)
        diff_x = torch.flatten(x_right - x_grid).unsqueeze(1)
        diff_y = torch.flatten(y_high - y_grid).unsqueeze(1)

        inter_feature = (feature11 * diff_x * diff_y +
                         feature12 * (1 - diff_x) * diff_y +
                         feature21 * diff_x * (1 - diff_y) +
                         feature22 * (1 - diff_x) * (1 - diff_y))

        inter_feature = inter_feature.view(batch_size, T, d_dim)
        interpolated_feature[:,l,:,:] = inter_feature

    interpolated_feature = torch.sum(interpolated_feature, dim=1)

    return interpolated_feature

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
                 hidden_size=128, leaky=False):
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

    def __init__(self, dim=3, z_dim=128, c_dim=32,
                 hidden_size=256):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.c_dim = c_dim

        #self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_p = nn.Linear(dim, c_dim)
        self.block0 = ResnetBlockFC(c_dim, c_dim)
        self.block1 = ResnetBlockFC(c_dim, c_dim)
        self.block2 = ResnetBlockFC(c_dim, c_dim)
        self.block3 = ResnetBlockFC(c_dim, c_dim)
        self.block4 = ResnetBlockFC(c_dim, c_dim)

        #self.fc_out = nn.Conv1d(c_dim, 1, 1)
        self.fc_out = nn.Linear(c_dim, 1)
        self.actvn = F.relu


        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")


    def forward(self, p, z, c, C_mat, **kwargs):

        batch_size, T, d = p.size()
        _, L, H, W, d_dim = c.size()

        # Billinear Interpolation
        max_dim = 0.55

        interval = float(2 / (H-1))
      
        c = BillinearInterpolation(p, c, C_mat, max_dim, interval)
        # c = BillinearInterpolationSphere(p, c, max_dim)
        #p = p.transpose(1, 2)
        net = self.fc_p(p)
        #net = self.actvn(net)
        #net = net.transpose(1, 2)
        net = net + c
        net = self.block0(net)
        net = net + c
        net = self.block1(net)
        net = net + c
        net = self.block2(net)
        net = net + c
        net = self.block3(net)
        net = net + c
        net = self.block4(net)        

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        
        return out

class DecoderSphere(nn.Module):
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
                 hidden_size=256, meridian = 64, latitude = 64):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.c_dim = c_dim
        self.meridian = meridian
        self.latitude = latitude

        #self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_p = nn.Linear(dim, c_dim)
        self.block0 = ResnetBlockFC(c_dim, c_dim)
        self.block1 = ResnetBlockFC(c_dim, c_dim)
        self.block2 = ResnetBlockFC(c_dim, c_dim)
        self.block3 = ResnetBlockFC(c_dim, c_dim)
        self.block4 = ResnetBlockFC(c_dim, c_dim)

        #self.fc_out = nn.Conv1d(c_dim, 1, 1)
        self.fc_out = nn.Linear(c_dim, 1)
        self.actvn = F.relu


        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")


    def forward(self, p, z, c, C_mat, **kwargs):

        batch_size, T, d = p.size()
        _, L, H, W, d_dim = c.size()

        # Billinear Interpolation
        max_dim = 0.55

        interval = float(2 / (H-1))
        #############changes!!!!!!!!!!!!!!!!!!!!###############
        # c = BillinearInterpolation(p, c, C_mat, max_dim, interval)
        c = BillinearInterpolationSphere(self.meridian, self.latitude, p, c, max_dim)
        #p = p.transpose(1, 2)
        net = self.fc_p(p)
        #net = self.actvn(net)
        #net = net.transpose(1, 2)
        net = net + c
        net = self.block0(net)
        net = net + c
        net = self.block1(net)
        net = net + c
        net = self.block2(net)
        net = net + c
        net = self.block3(net)
        net = net + c
        net = self.block4(net)        

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

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
