#### 3D Vision reimplementation
## Dusan, Daniil
import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
#from sklearn.neighbors import kneighbors_graph
import numpy as np
from numpy import linalg as LA

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

#### Another Encoder, enlisted under 'pointplanenet_resnet' in configuration

class PointPlaneResnet(nn.Module):
    ''' PointPlaneNet-based encoder network with ResNet blocks. \n
    Authors: 
        Daniil Emtsev and Dusan Svilarkovic 
    Args:
        c_dim (int): dimension of latent code c,  defined by config's model.c_dim = 512
        dim (int): input points dimension, in our case 
        hidden_dim (int): hidden dimension of the network
        k (int) : number of neighbours in grouping layer
        channels (int) : number of planes/channels
    '''

    def __init__(self, 
                c_dim=128, 
                dim=3, 
                hidden_dim=128, 
                k = 40,
                channels = 3):

        super().__init__()

        # #TODO comment this, it can be used from config
        hidden_dim = 512
        c_dim = 512

        # #end TODO

        self.c_dim = c_dim
        self.k = k #grouping layer
        self.channels = channels

     
        ##Parameters are Tensor subclasses, that have a very special property when used 
        # with Module s - when they’re 
        # assigned as Module attributes they are automatically added 
        # to the list of its parameters, and will appear e.g. in parameters() iterator. 
        self.plane_weights = torch.nn.Parameter(torch.randn(channels, 4).cuda())

        torch.nn.init.xavier_normal_(self.plane_weights)
        

        # self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.plane_conv = convolution
        
        self.mlp = MLP(channels = channels)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        
        self.actvn = nn.ReLU()
        self.pool = maxpool
        self.channels = channels
        self.weight = torch.nn.Parameter(torch.randn(channels, 4))
        self.weight.requires_grad = True

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        # net = self.fc_pos(p)

        # print(f'Weight planes {self.plane_weights}')

        net_batch = []
        for i in range(batch_size):
            # print(f'Weight planes {self.plane_weights}')

            net_sample = self.plane_conv(p[i,:,:], self.k, self.plane_weights, self.channels)
            net_batch.append(net_sample)
            # print(f'net_sample is {net_sample}')

        # print(f'net_batch: {net_batch}')

        net = torch.stack(net_batch)
        net = self.mlp(net)

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



##################new stuff , Numpy is the first part################################
# def knn_find(points, k: int):
#     """find indexes of k nearest points for every point in "points" array

#         Parameters
#         ----------
#         points   : np.array 
#                    set of 3d points [N*3]
#         k        : int
#                    number of nearest neighbours

#         Returns
#         -------
#         tensor [N*k] - for every point indexes of k nearest neighbours
#         """

#     graph = kneighbors_graph(
#         points, k, mode='connectivity', include_self=False)
#     array_neighbours = graph.toarray()

#     # return torch.from_numpy(array_neighbours)
#     return array_neighbours


# def kernel(weights, point):
#     """calculates H(W, X)
#     """

#     new_point = np.insert(point, 0, 1., axis=0)
#     answer = np.dot(weights, new_point)/LA.norm(weights[1:])
# #     print(answer)
#     return answer


# def final_kernel(points, i, weights, channel, k, id_neighbours):
#     """calculates 1/(1 + exp(H(W, X)))
#     """

#     pc_first = 1/k*sum([kernel(weights[channel, :], points[id_neighbour] - points[i])
#                         for id_neighbour in id_neighbours])
#     pc_final = 1/(1 + np.power(2.73, pc_first))
#     return pc_final


# def convolution(points, k, weights, channels):
#     """creates features for every point
#     Parameters
#         ----------
#         points   : torch.tensor
#                    set of 3d points [N*3]
#         k        : int
#                    number of nearest neighbours
#         weights  : torch.tensor
#                    set of weights [channels*4]
#         channels : int
#                    number of channels

#         Returns
#         -------
#         tensor [N*channels] - for every point "channels" features
    
#     """
#     points_array = points.numpy()  # torch tensor to np.array
#     weights = weights.numpy()
#     number_points = points.shape[0]
#     knn_array = knn_find(points_array, k)
#     array_features = np.zeros((number_points, channels))
#     for i in range(number_points):
#         id_neighbours = np.nonzero(knn_array[i])[0]
#         array_features[i] = np.asarray([final_kernel(
#             points_array, i, weights, channel, k, id_neighbours) for channel in np.arange(0, channels, 1)])
#     array_features = torch.from_numpy(array_features)
#     return array_features

##################new stuff################################
        def knn_find(points, k: int):
            """find indexes of k nearest points for every point in "points" array
                Parameters
                ----------
                points   : np.array 
                        set of 3d points [N*3]
                k        : int
                        number of nearest neighbours
                Returns
                -------
                tensor [N*k] - for every point indexes of k nearest neighbours
                """

            graph = kneighbors_graph(
                points, k, mode='connectivity', include_self=False)
            array_neighbours = graph.toarray()

            # return torch.from_numpy(array_neighbours)
            return array_neighbours


        def kernel(weights, point):
            """calculates H(W, X)
            """

            new_point = np.insert(point, 0, 1., axis=0)
            answer = np.dot(weights, new_point)/LA.norm(weights[1:])
        #     print(answer)
            return answer


        def final_kernel(points, i, weights, channel, k, id_neighbours):
            """calculates 1/(1 + exp(H(W, X)))
            """

            pc_first = 1/k*sum([kernel(weights[channel, :], points[id_neighbour] - points[i])
                                for id_neighbour in id_neighbours])
            pc_final = 1/(1 + np.power(2.73, pc_first))
            return pc_final


        def convolution(points, k, weights, channels):
            """creates features for every point
            Parameters
                ----------
                points   : torch.tensor
                        set of 3d points [N*3]
                k        : int
                        number of nearest neighbours
                weights  : torch.tensor
                        set of weights [channels*4]
                channels : int
                        number of channels
                Returns
                -------
                tensor [N*channels] - for every point "channels" features
            
            """
            points_array = points.numpy()  # torch tensor to np.array
            weights = weights.numpy()
            number_points = points.shape[0]
            knn_array = knn_find(points_array, k)
            array_features = np.zeros((number_points, channels))
            for i in range(number_points):
                id_neighbours = np.nonzero(knn_array[i])[0]
                array_features[i] = np.asarray([final_kernel(
                    points_array, i, weights, channel, k, id_neighbours) for channel in np.arange(0, channels, 1)])
            array_features = torch.from_numpy(array_features)
            return array_features




##############torch version######################
## Author : Daniil Emstev

def knn_find(points, k: int):
    """find indexes of k nearest points for every point in "points" array
        Parameters
        ----------
        points   : np.array 
                   set of 3d points [N*3]
        k        : int
                   number of nearest neighbours
        Returns
        -------
        tensor [N*k] - for every point indexes of k nearest neighbours
        """

    graph = kneighbors_graph(
        points, k, mode='connectivity', include_self=False)
    array_neighbours = graph.toarray()

    # return torch.from_numpy(array_neighbours)
    return array_neighbours


def kernel(weights, point):
    """calculates H(W, X)
    """
    new_point = torch.cat((torch.tensor([1.]).cuda(), point), dim=0)
    answer = weights.dot(new_point)/(weights[1:].norm(p=2))
    return answer


def final_kernel(points, i, weights, channel, k, id_neighbours):
    """calculates 1/(1 + exp(H(W, X)))
    """
#     print(weights[channel])
    pc_first = 1/k*sum([kernel(weights[channel], points[id_neighbour] - points[i])
                        for id_neighbour in id_neighbours if id_neighbour != i])
    # pc_final = 1/(1. + np.power(2.73, pc_first.numpy()))
    #Previous doesn't work for cuda 
    pc_final = 1/(1. + np.power(2.73, pc_first.item()))
    return pc_final


def convolution(points, k, weights, channels):
    """creates features for every point
    Author
        Daniil Emtsev
    Parameters
        ----------
        points   : torch.tensor
                   set of 3d points [N*3]
        k        : int
                   number of nearest neighbours
        weights  : torch.tensor
                   set of weights [channels*4]
        channels : int
                   number of channels
        Returns
        -------
        tensor [N*channels] - for every point "channels" features
    
    """
    number_points = points.shape[0]
    # array_features = torch.zeros([number_points, channels], dtype=torch.int32)
    # array_features = torch.zeros([number_points, channels], dtype=torch.float).cuda()
    array_features = []
    for i in range(number_points):
        dist = torch.norm(points - points[i], dim=1, p=None).cuda()
        #For PyTorch version 1.0.0  https://pytorch.org/docs/1.0.0/torch.html?highlight=topk#torch.topk
        id_neighbours = dist.topk(k+1, largest=False)[1]
        array_feature = torch.tensor([final_kernel(
            points, i, weights, channel, k, id_neighbours) for channel in np.arange(0, channels, 1)],dtype=torch.float).cuda()
        # array_feature = [final_kernel(
        #     points, i, weights, channel, k, id_neighbours) for channel in np.arange(0, channels, 1)]
        array_features.append(array_feature)
    array_features = torch.stack(array_features).cuda()

    return array_features


### Dusan's Stuff

class MLP(nn.Module):
    """
    From Point-PlaneNet : Then 4 MLP layers (respectfully with 64, 128, 128, 1024 neurons) are used \n
    to transform per-point’s features to higher-dimensional space and a global \n
    max pooling layer is used to extract global feature vector. \n
    All MLPs are followed by batch normalization and ReLU. \n
    Input:
        size N x L
    Output: 
        size N x 1024
    Author : 
        Dusan Svilarkovic
    Parameters :
        channels (int) : number of planes/channels
    """
    def __init__(self, channels = 3):
        
        super(MLP, self).__init__()
        self.channels = channels
        self.mlp64 = nn.Linear(channels, 64)
        self.batchnorm64 = nn.BatchNorm1d(64)
        
        self.mlp128_1 = nn.Linear(64, 128)
        self.batchnorm128_1 = nn.BatchNorm1d(128)
        
        self.mlp128_2 = nn.Linear(128, 128)
        self.batchnorm128_2 = nn.BatchNorm1d(128)
        
        self.mlp1024 = nn.Linear(128, 1024)
        self.batchnorm1024 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

        
        

    def forward(self, x):
        
        #for batch processing
        output = []
        
        if(len(x.shape) == 3):
            for i in range(x.shape[1]):
                output_part = self.forward_function(x[:,i,:])
                output.append(output_part)

        
        output = torch.stack(output)
        output = output.permute(1,0,2)
        return(output)
        
        
    def forward_function(self, x):
        x = self.mlp64(x)
        x = self.batchnorm64(x)
        x = self.relu(x)
        
        x = self.mlp128_1(x)
        x = self.batchnorm128_1(x)
        x = self.relu(x)
        
        x = self.mlp128_2(x)
        x = self.batchnorm128_2(x)
        x = self.relu(x)
        
        x = self.mlp1024(x)
        
        
        return x
