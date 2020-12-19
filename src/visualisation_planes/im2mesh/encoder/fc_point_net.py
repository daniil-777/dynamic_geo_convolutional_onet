#### 3D Vision reimplementation
## Dusan, Daniil, Stefan
import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC, Unet, FCPlanenet
import numpy as np
from numpy import linalg as LA
from torch_scatter import scatter_mean, scatter_max
import os
import re

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out
        
def ChangeBasis(plane_parameters, device = 'cuda'):
    # Input: Plane parameters (batch_size x L x 3) - torch.tensor dtype = torch.float32
    # Output: C_mat (batch_size x L x 4 x 3)
    # C_mat is stacked matrices of:
    # 1. Change of basis matrices (batch_size x L x 3 x 3)
    # 2. Normalizing constants (batch_size x L x 1 x 3)
    device = device

    batch_size, L, _ = plane_parameters.size()
    normal = plane_parameters.reshape([batch_size * L, 3]).float()
    normal = normal / torch.norm(normal, p=2, dim=1).view(batch_size * L, 1) #normalize
    normal = normal + 0.0001 # Avoid non-invertible matrix down the road

    basis_x = torch.tensor([1, 0, 0], dtype=torch.float32).repeat(batch_size*L,1).to(device)
    basis_y = torch.tensor([0, 1, 0], dtype=torch.float32).repeat(batch_size*L,1).to(device)
    basis_z = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(batch_size*L,1).to(device)

    v = torch.cross(basis_z.to(device), normal)
    zero = torch.zeros([batch_size*L], dtype=torch.float32).to(device)
    skew = torch.zeros([batch_size*L, 3, 3], dtype=torch.float32).to(device)
    skew[range(batch_size*L), 0] = torch.stack([zero, -v[:,2], v[:,1]]).t()
    skew[range(batch_size * L), 1] = torch.stack([v[:,2], zero, -v[:,0]]).t()
    skew[range(batch_size * L), 2] = torch.stack([-v[:,1], v[:,0], zero]).t()

    idty = torch.eye(3).to(device)
    idty = idty.reshape((1, 3, 3))
    idty = idty.repeat(batch_size*L, 1, 1)
    dot = (1-torch.sum(normal*basis_z,dim=1)).unsqueeze(1).unsqueeze(2)
    div = torch.norm(v, p=2, dim=1)**2
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
    p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32).repeat(batch_size*L,1).to(device)
    p_x = torch.sum(b_x*p_dummy,dim=1).unsqueeze(1) / torch.sum(b_x*b_x,dim=1).unsqueeze(1) * b_x
    p_y = torch.sum(b_y*p_dummy,dim=1).unsqueeze(1) / torch.sum(b_y*b_y,dim=1).unsqueeze(1)* b_y

    c_x = torch.norm(p_x, p=2, dim=1)
    c_y = torch.norm(p_y, p=2, dim=1)

    normalizer = torch.max(c_x, c_y).unsqueeze(1).unsqueeze(2).repeat(1,1,3)

    C_mat = torch.cat([C_inv, normalizer], dim=1)

    C_mat = C_mat.view(batch_size,L,4,3)

    return C_mat


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network. With plane training

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, n_channels = 3):
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
   #  PointNet-based encoder network with ResNet blocks.

   # Args:
   #     c_dim (int): dimension of latent code c
   #     dim (int): input points dimension
   #     hidden_dim (int): hidden dimension of the network
   #     n_channels (int): number of planes for projection
    

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, n_channels = 4, plane_param_file="normals.csv", object="null"):
        super().__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim   
        self.n_channels = n_channels

        self.file_name = str(plane_param_file)
        self.object = str(object)
        
        # For grid features
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.unet = Unet(hidden_dim)

        # For plane prediction
        self.fc_plane_net = FCPlanenet(n_dim=dim, n_channels=n_channels, hidden_dim=hidden_dim)
        self.fc_plane_hdim = nn.Linear(n_channels*3, hidden_dim)

        # Activation & pooling
        self.actvn = nn.ReLU()
        self.pool = maxpool
        
        is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")

    def forward(self, p):
        batch_size, T, D = p.size()
        

        # Grid features
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
        net = self.block_4(net) # batch_size x T x hidden_dim (T: number of sampled input points)

        net_pl = self.fc_plane_net(p)
        plane_parameters = net_pl.view(batch_size,-1,3) # batch_size x L x 3

        flatten_tensor = torch.flatten(plane_parameters)
        string = str(flatten_tensor)
        string = string.replace('\n', ' ').replace(' ', '')
        sliced_string = re.findall(r'\[(.*?)\]', string)[0]
        st = self.object + "," + sliced_string + "\n"

        if os.path.isfile(self.file_name) == False:
            log = open(self.file_name, "w+")
            log.write(st)
            print("{} normals appended to {}".format(self.object, self.file_name))
            log.close()
        else:
            log = open(self.file_name, "a")
            log.write(st)
            print("{} normals appended to {}".format(self.object, self.file_name))
            log.close()

        raise ValueError('Intentionally stop program here. No need to run full generation.')


        #eye_basis = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(self.device)
        #canonical_planes = torch.cat(batch_size*[eye_basis]).view(batch_size, 3, 3)
        #plane_parameters = canonical_planes

        C_mat = ChangeBasis(plane_parameters, device = self.device)  # batch_size x L x 4 x 3
        net_pl = self.fc_plane_hdim(self.actvn(net_pl))
        net_pl = net_pl.unsqueeze(1) # batch_size x 1 x hidden_dim

        # Combine net and net_pl
        net = net + net_pl # to allow backpropagation to net_pl        

        # Create grid feature
        grid_res = 64
        max_dim = 0.55
        H = grid_res
        W = grid_res
        interval = float(2 / (grid_res-1))

        c = torch.zeros([batch_size, self.n_channels, W, H, self.hidden_dim], device=self.device)

        for l in range(C_mat.size()[1]):
            p_project = torch.div(p, max_dim)
            p_project = torch.transpose(p_project, 2,1)
            p_project = torch.bmm(C_mat[:,l,:3], p_project)
            p_project = torch.transpose(p_project, 2,1)

            p_project = p_project / (C_mat[:,l,3,0]+0.05).unsqueeze(1).unsqueeze(2) # divide by normalizer so that range is [-1,1]
            p_project = p_project[:,:,:2]
            xy_index = (p_project + 1) / interval
            xy_index[xy_index>=(grid_res-1)] = grid_res-1-0.1
            xy_index[xy_index<0] = 0
            xy_index = torch.round(xy_index).int()
            cell_index = xy_index[:,:,0] + H * xy_index[:,:,1]
            cell_index = cell_index.unsqueeze(2).long()
            out = net.new_zeros((batch_size, W*H, self.hidden_dim)).to(self.device)
            out, _ = scatter_max(net, cell_index, dim=1, out=out)
            c[:,l,] = out.view(batch_size, H, W, self.hidden_dim)


        # Reshape for U-Net
        _, L, H, W, d_dim = c.size()
        c = c.view([batch_size * L, H, W, d_dim])
        c = c.permute(0, 3, 1, 2)

        # U-Net
        c = self.unet(c)
        c = c.permute(0, 2, 3, 1)
        c = c.view(batch_size, L, H, W, self.c_dim)

        return c, C_mat
