import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from im2mesh.dvr_common import normalize_imagenet


# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, z_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.c_dim = c_dim

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 512)

        self.fc_mean = nn.Linear(512, z_dim)
        self.fc_logstd = nn.Linear(512, z_dim)

    def forward(self, x, c=None):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        if self.c_dim != 0:
            net_c = self.fc_c(c)
            net = net + net_c
        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(dim, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        self.fc_0 = nn.Linear(1, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, p, x, c=None, **kwargs):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_0(x.unsqueeze(-1))
        net = net + self.fc_pos(p)

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce
        #  to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
