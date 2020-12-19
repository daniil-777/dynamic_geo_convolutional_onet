import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from math import sin,cos,radians,sqrt
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, DEGREES = 0):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, DEGREES = DEGREES)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data, DEGREES = 0):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if(DEGREES != 0):
            inputs, rotation = self.rotate_points(inputs, DEGREES = DEGREES)
            p = self.rotate_points(p, use_rotation_tensor = True)

        kwargs = {}

        c, C_mat = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, C_mat, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss



    def rotate_points(self, pointcloud_model, DEGREES = 0, query_points = False, use_rotation_tensor = False, save_rotation_tensor = False):
            ## https://en.wikipedia.org/wiki/Rotation_matrix
        """
            Function for rotating points
                pointcloud_model (numpy 3d array) - batch_size x pointcloud_size x 3d channel sized numpy array which presents pointcloud
                DEGREES (int) - range of rotations to be used
                query_points (boolean) - used for rotating query points with already existing rotation matrix
                use_rotation_tensor (boolean) - asking whether DEGREES should be used for generating new rotation matrix, or use the already established one
                save_rotation_tensor (boolean) - asking to keep rotation matrix in a pytorch .pt file
        """
        if(use_rotation_tensor != True):
            angle_range = DEGREES
            x_angle = radians(random.random() * angle_range)
            y_angle = radians(random.random() * angle_range)
            z_angle = radians(random.random() * angle_range)

            # print(x_angle,y_angle, z_angle)
            rot_x = torch.Tensor([[1,0,0,0],[0, cos(x_angle),-sin(x_angle),0], [0, sin(x_angle), cos(x_angle),0], [0,0,0,1]])
            rot_y = torch.Tensor([[cos(y_angle),0,sin(y_angle), 0],[0, 1, 0,0], [-sin(y_angle),0,cos(y_angle),0], [0,0,0,1]])
            rot_z = torch.Tensor([[cos(z_angle), -sin(z_angle),0,0],[sin(z_angle), cos(z_angle),0,0],[0,0,1,0], [0,0,0,1]])
            rotation_matrix = torch.mm(rot_y, rot_z)
            rotation_matrix = torch.mm(rotation_matrix, rot_x)        

            # pointcloud_model = pointcloud_model[0,:,:]
            # pointcloud_model = pointcloud_model / sqrt(3)
        
            pointcloud_model = pointcloud_model / sqrt(0.55**2 + 0.55**2 + 0.55**2) # / sqrt(0.75)
            batch_size, point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = torch.cat([pointcloud_model, torch.ones(batch_size, point_cloud_size,1).to(self.device)], dim = 2)
            
    
            pointcloud_model_rotated = torch.matmul(pointcloud_model, rotation_matrix.to(self.device))
            self.rotation_matrix = rotation_matrix
            
            if(save_rotation_tensor):
                torch.save(rotation_matrix, '/home/dsvilarkovic/partition600/rotation_matrix.pt') #used for plane prediction
            return pointcloud_model_rotated[:,:,0:3], (x_angle, y_angle, z_angle)
        else: 
            batch_size, point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = pointcloud_model / sqrt(0.55**2 + 0.55**2 + 0.55**2)
            pointcloud_model = torch.cat([pointcloud_model, torch.ones(batch_size, point_cloud_size,1).to(self.device)], dim = 2)
            pointcloud_model_rotated =torch.matmul(pointcloud_model, self.rotation_matrix.to(self.device))
            return pointcloud_model_rotated[:,:,0:3]
