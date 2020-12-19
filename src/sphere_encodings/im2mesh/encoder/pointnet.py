import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC, Unet
import numpy as np

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def SingleChangeBasisMatrix(normal):
    # single_plane_parameter - torcch.tensor[1*4] dtype = torch.float32
    # a, b, c, _ = single_plane_parameter[0],
    # a, b, c = float(a), float(b), float(c)
    normal = normal.float()
    normal = normal / torch.sqrt(torch.sum(normal ** 2))

    if sum(normal == torch.tensor([0, 0, 1], dtype=torch.float32)) != torch.tensor(
        3, dtype=torch.uint8
    ):
        basis_x = torch.tensor([1, 0, 0], dtype=torch.float32)
        basis_y = torch.tensor([0, 1, 0], dtype=torch.float32)
        basis_z = torch.tensor([0, 0, 1], dtype=torch.float32)

        # Construct rotation matrix to align z-axis basis to plane normal
        # Need to add exception, if normal = [0, 0, 1]. don't do basis rotation
        v = torch.cross(basis_z, normal)
        #         print(v[0].view(-1).shape)
        zero_tensor = torch.tensor(0, dtype=torch.float32)
        ssc = torch.tensor(
            [
                zero_tensor,
                -v[2],
                v[1],
                v[2],
                zero_tensor,
                -v[0],
                -v[1],
                v[0],
                zero_tensor,
            ]
        ).view(3, 3)
        R = (
            torch.eye(3)
            + ssc
            + torch.matmul(ssc, ssc)
            * (1 - torch.dot(normal, basis_z))
            / (torch.norm(v, p=2, dim=0) ** 2)
        )

        # Change basis to plane normal basis
        # plane equation in new basis: z = 0
        # plane normal basis in standard coordinate
        new_basis_x = torch.matmul(R, basis_x)
        new_basis_y = torch.matmul(R, basis_y)
        new_basis_z = torch.matmul(R, basis_z)
        b_x = torch.abs(new_basis_x).view(-1)
        b_y = torch.abs(new_basis_y).view(-1)
        p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32)
        p_x = torch.dot(p_dummy, b_x) / torch.dot(b_x, b_x) * b_x
        p_y = torch.dot(p_dummy, b_y) / torch.dot(b_y, b_y) * b_y
        c_x = torch.norm(p_x, p=2, dim=0)
        c_y = torch.norm(p_y, p=2, dim=0)

        if c_x > c_y:
            norm_c = torch.tensor([c_x, c_x, c_x])
        else:
            norm_c = torch.tensor([c_y, c_y, c_y])
        # really cat wrt 1 dim?

        new_basis_matrix = torch.t(
            torch.cat(
                (
                    torch.transpose(new_basis_x, 0, -1),
                    torch.transpose(new_basis_y, 0, -1),
                    torch.transpose(new_basis_z, 0, -1),
                ),
                0,
            ).view(3, 3)
        )
        #print("new_basis_shape", new_basis_matrix.shape)
        C_inv = torch.inverse(new_basis_matrix)
        C_inv = C_inv.contiguous().view(-1)

    else:
        C_inv = torch.eye(3).view(-1)
        norm_c = torch.ones((3,))

    C_inv_norm_c = torch.cat((C_inv, norm_c), dim=0).view(4, 3)

    return C_inv_norm_c


def ChangeBasisMatrix(plane_parameters):
    # Input: Plane parameters (Lx4) - torch.tensor dtype = torch.float32
    # Output: Change of basis matrices (L x 3 x 3)
    L = plane_parameters.shape[0]
    mat = SingleChangeBasisMatrix(plane_parameters[0])

    for i in range(1, L):
        # really cat wrt 0 dim?
        mat = torch.cat((mat, SingleChangeBasisMatrix(plane_parameters[i])), 0)
    mat = mat.view(L, 4, 3).to('cuda')

    return mat

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
        
        # Reducee to  B x F
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

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
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
        self.unet = Unet(hidden_dim)

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

        # Assume have plane parameters of size Lx3
        plane_param_np = np.array([[1, 1, 1], [1,0,0], [0,1,0], [0,0,1]])
        plane_parameters = torch.tensor(plane_param_np)

        C_mat = ChangeBasisMatrix(plane_parameters)  # L x 3 x 3

        # Create grid feature
        grid_res = 64
        max_dim = 0.55
        H = grid_res
        W = grid_res
        interval = float(2 / (grid_res-1))

        c = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')
        counter = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]], device='cuda')
        #c = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]])
        #counter = torch.zeros([net.size()[0], C_mat.size()[0], W, H, net.size()[2]])

        for l in range(C_mat.size()[0]):
            p_project = torch.div(p, max_dim)
            proj_mat = C_mat[l][:3,].t()
            p_project = torch.matmul(p_project, proj_mat)

            norm_c = C_mat[l][3,0]
            p_project = p_project[:, :, 0:2] / (norm_c+0.01)  # divide by norm_c so that range is [-1,1]
            xy_index = torch.round((p_project + 1) / interval).int()

            for n in range(p.size()[1]):
                x_grid, y_grid = xy_index[:, n, 0], xy_index[:, n, 1]
                x_grid = x_grid.tolist()
                y_grid = y_grid.tolist()
                counter[range(p.size(0)), l, x_grid, y_grid] = counter[range(p.size(0)), l, x_grid, y_grid] + 1
                c[range(p.size(0)), l, x_grid, y_grid] = c[range(p.size(0)), l, x_grid, y_grid] + \
                                                                    net[range(p.size(0)), n]

        # Average overlapping projection
        counter[counter == 0] = 1
        c = torch.div(c, counter)

        # Reshape for U-Net
        _, L, H, W, d_dim = c.size()
        c = c.reshape([batch_size * L, H, W, d_dim])
        c = c.permute(0, 3, 1, 2)

        # U-Net
        c = self.unet(c)
        c = c.permute(0, 2, 3, 1)
        c = c.view(batch_size, L, H, W, self.c_dim)

        return c, C_mat
