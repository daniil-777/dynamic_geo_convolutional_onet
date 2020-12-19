#### Point Plane Net reimplementation of 
#https://www.sciencedirect.com/science/article/pii/S1051200419301873?via%3Dihub
##Daniil, Dusan
import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
from sklearn.neighbors import kneighbors_graph
import numpy as np
from numpy import linalg as LA

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out





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