'''
From the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
'''

import torch.nn as nn
import torch.nn.functional as F
from im2mesh.encoder.pointnetpp_util import PointNetSetAbstractionMsg,PointNetFeaturePropagation, PointNetSetAbstraction
import pdb


# class PointNetPlusPlus(nn.Module):
#     def __init__(self, c_dim=128, padding=0.1):
#         super(PointNetPlusPlus, self).__init__()
#
#         self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
#         self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
#         self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
#         self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
#         self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
#         self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(128, [128, 128, c_dim])
#         # self.conv1 = nn.Conv1d(128, 128, 1)
#         # self.bn1 = nn.BatchNorm1d(128)
#         # self.drop1 = nn.Dropout(0.5)
#         # self.conv2 = nn.Conv1d(128, num_classes, 1)
#
#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)
#         l0_points = xyz
#         l0_xyz = xyz[:,:3,:]
#
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#
#         l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
#
#         # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
#         # x = self.conv2(x)
#         # x = F.log_softmax(x, dim=1)
#         # x = x.permute(0, 2, 1)
#         return xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1)

class PointNetPlusPlus(nn.Module):
    def __init__(self, c_dim=128, padding=0.1):
        super(PointNetPlusPlus, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, c_dim])

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1)


if __name__ == '__main__':
    import  torch
    model = get_model(13)
    pdb.set_trace()
    xyz = torch.rand(6, 3, 2048)
    (model(xyz))
