import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class FCPlanenet(nn.Module):
    """
    For reducing point clouds we use SimplePointNet to reduce to c_dim features,
    and on it we define our FC to derive planes we need.
    Input:
        size N x n_dim
    Output: 
        size n_channels x n_dim
    Author : 
        Dusan Svilarkovic
    Parameters :
        n_channels (int) : number of planes/channels
        n_dim (int) : dimension of points (3 for 3D, 2 for 2D)
        n_points (int) : number of points
        c_dim (int) : dimension of out
    """
    def __init__(self, 
                n_dim = 3, 
                n_channels = 3, 
                n_points = 300,
                c_dim = 32,
                hidden_dim = 32,
                train_normals_flag = "rotation_matrix"):
        super(FCPlanenet, self).__init__()

        # Simple PointNet
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(n_dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        # MLP
        #uses c_dim = hidden_dim to give accordingly proper output for further reducing to plane parameters
        #self.point_net = SimplePointnet(c_dim=hidden_dim,hidden_dim=hidden_dim, dim=n_dim)

        self.mlp0 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        #converts point cloud to its n_channels plane that best explains it
        self.to_planes = nn.Linear(hidden_dim, n_channels * 3)
        #in case of learning matrix rotation
  
         
    def forward(self, p):
        batch_size, T, D = p.size()

        # Simple Point Net
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
        
        # Reduce to B x hidden_dim
        net = self.pool(net, dim=1)
        net = self.fc_c(self.actvn(net))

        # MLP
        net = self.mlp0(self.actvn(net))
        net = self.mlp1(self.actvn(net))
        net = self.mlp2(self.actvn(net))
     
        net = self.to_planes(self.actvn(net))
        
        return(net)


class FCRotationPlanenet(nn.Module):
    """
    For reducing point clouds we use SimplePointNet to reduce to c_dim features,
    and on it we define our FC to derive planes we need.
    Input:
        size N x n_dim
    Output: 
        size n_channels x n_dim
    Author : 
        Daniil Emtsev
    Parameters :
        n_channels (int) : number of planes/channels
        n_dim (int) : dimension of points (3 for 3D, 2 for 2D)
        n_points (int) : number of points
        c_dim (int) : dimension of out
    """
    def __init__(self, 
                n_dim = 3, 
                n_channels = 3, 
                n_points = 300,
                c_dim = 32,
                hidden_dim = 32,):
        super(FCRotationPlanenet, self).__init__()

        # Simple PointNet
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(n_dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        # MLP
        #uses c_dim = hidden_dim to give accordingly proper output for further reducing to plane parameters
        #self.point_net = SimplePointnet(c_dim=hidden_dim,hidden_dim=hidden_dim, dim=n_dim)

        self.mlp0 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        #converts point cloud to its n_channels plane that best explains it
        self.to_planes = nn.Linear(hidden_dim, n_channels * 3)
        #in case of learning matrix rotation
        self.to_matrix_rotation = nn.Linear(hidden_dim, n_channels * 9)

         
    def forward(self, p):
        batch_size, T, D = p.size()

        # Simple Point Net
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

        # Reduce to B x hidden_dim
        net = self.pool(net, dim=1)
        net = self.fc_c(self.actvn(net))

        # MLP
        net = self.mlp0(self.actvn(net))
        net = self.mlp1(self.actvn(net))
        net = self.mlp2(self.actvn(net))
 
        net = self.to_matrix_rotation(self.actvn(net)) #in case of matrix rotation [batch, channels, 9]
            
        
        return(net)


class FCUniversalPlanenet(nn.Module):
    """
    For reducing point clouds we use SimplePointNet to reduce to c_dim features,
    and on it we define our FC to derive planes we need.
    Input:
        size N x n_dim
    Output: 
        size n_channels x n_dim
    Author : 
        Daniil Emtsev
    Parameters :
        n_channels (int) : number of planes/channels
        n_dim (int) : dimension of points (3 for 3D, 2 for 2D)
        n_points (int) : number of points
        c_dim (int) : dimension of out
    """
    def __init__(self, 
                n_dim = 3, 
                n_channels = 3, 
                n_points = 300,
                c_dim = 32,
                hidden_dim = 32,
                train_normals_flag = "rotation_matrix"):
        super(FCPlanenet, self).__init__()

        # Simple PointNet
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(n_dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        # MLP
        #uses c_dim = hidden_dim to give accordingly proper output for further reducing to plane parameters
        #self.point_net = SimplePointnet(c_dim=hidden_dim,hidden_dim=hidden_dim, dim=n_dim)

        self.mlp0 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        #converts point cloud to its n_channels plane that best explains it
        self.to_planes = nn.Linear(hidden_dim, n_channels * 3)
        #in case of learning matrix rotation
        self.to_matrix_rotation = nn.Linear(hidden_dim, n_channels * 9)
        self.train_normals_flag = train_normals_flag
         
    def forward(self, p):
        batch_size, T, D = p.size()

        # Simple Point Net
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

        # Reduce to B x hidden_dim
        net = self.pool(net, dim=1)
        net = self.fc_c(self.actvn(net))

        # MLP
        net = self.mlp0(self.actvn(net))
        net = self.mlp1(self.actvn(net))
        net = self.mlp2(self.actvn(net))
        if (self.train_normals_flag == "rotation_matrix"):
            net = self.to_matrix_rotation(self.actvn(net)) #in case of matrix rotation [batch, channels, 9]
            
        else:
            net = self.to_planes(self.actvn(net))
        
        return(net)



class DoubleConv_LeftU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == out_channels:
            mid_channels = in_channels
            out_channels = mid_channels
        else:
            mid_channels = in_channels * 2
            out_channels = mid_channels
        self.double_conv_left = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2) #for bigger kernel size
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_left(x)

class DoubleConv_RightU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == out_channels:
            mid_channels = in_channels
            out_channels = mid_channels
        else:
            mid_channels = int(in_channels/2)
            out_channels = mid_channels
        self.double_conv_right = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2) #for bigger kernel size
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_right(x)


class DownSampling_DoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_LeftU(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Resnet Blocks
class Unet(nn.Module):
    ''' U-net class.

    Args:
        d_dim: depth of features
    '''

    def __init__(self, d_dim):
        super().__init__()
        # Submodules
        self.DoubleConv2dL0 = DoubleConv_LeftU(d_dim, d_dim)
        self.Down1 = DownSampling_DoubleConv(d_dim, d_dim * 2)
        self.Down2 = DownSampling_DoubleConv(d_dim * 2, d_dim * 4)
        self.Down3 = DownSampling_DoubleConv(d_dim * 4, d_dim * 8)
        self.Down4 = DownSampling_DoubleConv(d_dim * 8, d_dim * 16)
        self.Up4 = nn.ConvTranspose2d(d_dim * 16, d_dim * 8, kernel_size=2, stride=2)
        self.DoubleConv2dR3 = DoubleConv_RightU(d_dim * 16, d_dim * 8)
        self.Up3 = nn.ConvTranspose2d(d_dim * 8, d_dim * 4, kernel_size=2, stride=2)
        self.DoubleConv2dR2 = DoubleConv_RightU(d_dim * 8, d_dim * 4)
        self.Up2 = nn.ConvTranspose2d(d_dim * 4, d_dim * 2, kernel_size=2, stride=2)
        self.DoubleConv2dR1 = DoubleConv_RightU(d_dim * 4, d_dim * 2)
        self.Up1 = nn.ConvTranspose2d(d_dim * 2, d_dim, kernel_size=2, stride=2)
        self.DoubleConv2dR0 = DoubleConv_RightU(d_dim * 2, d_dim)

    def forward(self, x):
        net0 = self.DoubleConv2dL0(x)
        net1 = self.Down1(net0)
        net2 = self.Down2(net1)
        net3 = self.Down3(net2)
        net = self.Down4(net3)

        net = self.Up4(net)
        net = torch.cat([net3, net], dim=1)
        net = self.DoubleConv2dR3(net)

        net = self.Up3(net)
        net = torch.cat([net2, net], dim=1)
        net = self.DoubleConv2dR2(net)

        net = self.Up2(net)
        net = torch.cat([net1, net], dim=1)
        net = self.DoubleConv2dR1(net)

        net = self.Up1(net)
        net = torch.cat([net0, net], dim=1)
        net = self.DoubleConv2dR0(net)

        return net

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert(x.size(0) == p.size(0))
        assert(p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out
