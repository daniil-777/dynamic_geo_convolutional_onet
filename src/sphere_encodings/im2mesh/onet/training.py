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

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
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

    def triplet_loss_normals(self, normals, margin):
        '''Calculates triplet loss of cos similarities among normals
           min(abs(cos_similarity) - margin, 0)
           where cos_similarity = sum[cos(normal_{i}, normal_{j})]
                                  for i in [o, num_normals], j in [i, num_normals]
        
        Author : 
        Daniil Emtsev

        Parameters
        ----------
        normals: torch.tensor 
                 shape [Batch, Num_normals, 3]
        margin: float
        
        Returns
        ----------
        losses: torch.tensor
                shape [Batch, C_{Num_normals}^{2}]  
                C_{Num_normals}^{2} - binom coefficient (number of combination of 2 from Num_normals elements)
        '''
        batch, T, dim = normals.size()
        similarity_list = []
        for i in range(T):
            for j in range(i + 1, T):
                similarity = torch.abs(torch.cosine_similarity(normals[:,i,:], normals[:,j,:])).unsqueeze(0)
        #         print(similarity.shape)
                similarity_list.append(similarity)
        sim_tensor = torch.cat(similarity_list)
        sim_tensor = torch.transpose(sim_tensor, 0, 1)
        # print("shape before tr", sim_tensor.shape)
        sim_tensor = torch.sum(sim_tensor, dim = 1).unsqueeze(1)
        # print(sim_tensor.shape)
        zeros = torch.zeros(batch).unsqueeze(1).to(self.device)
        losses = torch.min(sim_tensor - margin, zeros)
        return losses

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}
        # print("imputs shape", inputs.shape)
        # c, C_mat, plane_normals = self.model.encode_inputs(inputs) #uncomment in case of triplet loss
        c,  C_mat = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        
        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()
        # print("current loss shape", loss.shape)
        # print("curretn loss", loss)
        # General points
        logits = self.model.decode(p, z, c, C_mat, **kwargs).logits
        # print("logits shape", logits.shape)
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        # print("loss_i shape", loss_i.shape)
        # print("loss_i ", loss_i)

        # triplet_loss_normals = self.triplet_loss_normals(plane_normals, 1) #uncomment in case of triplet loss

        # print("triplet_loss", triplet_loss_normals)
        # print("trip_loss_mean", triplet_loss_normals.sum(-1).mean())
        loss = loss + loss_i.sum(-1).mean() #uncomment in case of triplet loss
        # loss = loss + loss_i.sum(-1).mean() + triplet_loss_normals.sum(-1).mean()

        
        # print("final loss shape", loss.shape)
        # print("final loss", loss)
        return loss
