import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.dvr.models import (
    encoder_latent, decoder, depth_function, texture_field
)
from im2mesh.dvr_common import (
    get_mask, image_points_to_world, origin_to_world, normalize_tensor)

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
    'resnet18': encoder_latent.Resnet18,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'batchnorm': decoder.DecoderBatchNorm,
    'local': decoder.LocalDecoder,
}

texture_field_dict = {
    'simple': texture_field.TextureField,
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, texture_field=None, encoder=None,
                 encoder_latent=None, depth_predictor=None, p0_z=None,
                 device=None, depth_function_kwargs={}):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)

        if texture_field is not None:
            self.texture_field = texture_field.to(device)

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        if depth_predictor is not None:
            self.depth_predictor = depth_predictor.to(device)
        else:
            self.depth_predictor = None

        self._device = device
        self.p0_z = p0_z

        self.call_depth_function = depth_function.DepthModule(
            **depth_function_kwargs)

    def forward(self, pixels, p_occupancy, p_freespace, inputs, camera_mat,
                world_mat, scale_mat, it=None,  sparse_depth=None,
                calc_normals=False, p_iou=None, **kwargs):
        ''' Performs a forward pass through the network.

        This function evaluate the depth and RGB color values for respective
        points as well as the occupancy values for the points of the helper
        losses. By wrapping everything in the forward pass, multi-GPU training
        is enabled.

        Args:
            pixels (tensor): sampled pixels
            p_occupancy (tensor): points for occupancy loss
            p_freespace (tensor): points for freespace loss
            inputs (tensor): input
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            it (int): training iteration (used for ray sampling scheduler)
            sparse_depth (dict): if not None, dictionary with sparse depth data
            calc_normals (bool): whether to calculate normals for surface
                points and a randomly-sampled neighbor
        '''
        # encode inputs
        c = self.encode_inputs(inputs)
        q_z = self.infer_z(inputs, c)
        z = q_z.rsample()

        # transform pixels p to world
        p_world, mask_pred, mask_zero_occupied = \
            self.pixels_to_world(pixels, camera_mat, world_mat, scale_mat, z,
                                 c, it)
        # if torch.any(torch.isnan(p_world)):
        #     import ipdb; ipdb.set_trace()
        #rgb_pred = self.decode_color(p_world, z=z, c=c)
        rgb_pred = None
        # eval occ at sampled p
        logits_occupancy = self.decode(p_occupancy, c=c, z=z).logits

        # eval freespace at p and
        # fill in predicted world points
        p_freespace[mask_pred] = p_world[mask_pred].detach()
        #p_freespace[mask_zero
        origin_world = origin_to_world(pixels.shape[1], camera_mat, world_mat, scale_mat)
        p_freespace[mask_zero_occupied] = origin_world[mask_zero_occupied]
        logits_freespace = self.decode(p_freespace, c=c, z=z).logits

        if calc_normals:
            normals = self.get_normals(p_world.detach(), mask_pred, z=z, c=c)
        else:
            normals = None

        # Project pixels for sparse depth loss to world if dict is not None
        if sparse_depth is not None:
            p = sparse_depth['p']
            camera_mat = sparse_depth['camera_mat']
            world_mat = sparse_depth['world_mat']
            scale_mat = sparse_depth['scale_mat']
            p_world_sparse, mask_pred_sparse, _ = self.pixels_to_world(
                p, camera_mat, world_mat, scale_mat, z, c, it)
        else:
            p_world_sparse, mask_pred_sparse = None, None

        if p_iou is not None:
            logits_iou = self.decode(p_iou, z=z, c=c).logits
        else:
            logits_iou = None

        return (p_world, rgb_pred, logits_occupancy, logits_freespace,
                mask_pred, p_world_sparse, mask_pred_sparse, normals, mask_zero_occupied, logits_iou)

    def get_normals(self, points, mask, z=None, c=None, h_sample=1e-3,
                    h_finite_difference=1e-3):
        ''' Returns the unit-length normals for points and one randomly
        sampled neighboring point for each point.

        Args:
            points (tensor): points tensor
            mask (tensor): mask for points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            h_sample (float): interval length for sampling the neighbors
            h_finite_difference (float): step size finite difference-based
                gradient calculations
        '''
        device = self._device

        if mask.sum() > 0:
            # if not (type(c) == dict):
            #     c = c.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
            # if not (type(z) == dict):
            #     z = z.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
            #points = points[mask]
            points_neighbor = points + (torch.rand_like(points) * h_sample -
                                        (h_sample / 2.))

            normals_p = normalize_tensor(
                self.get_central_difference(points, z=z, c=c,
                                            h=h_finite_difference))[mask]
            normals_neighbor = normalize_tensor(
                self.get_central_difference(points_neighbor, z=z, c=c,
                                            h=h_finite_difference))[mask]
        else:
            normals_p = torch.empty(0, 3).to(device)
            normals_neighbor = torch.empty(0, 3).to(device)

        return [normals_p, normals_neighbor]

    def get_central_difference(self, points, z=None, c=None, h=1e-3):
        ''' Calculates the central difference for points.

        It approximates the derivative at the given points as follows:
            f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

        Args:
            points (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            h (float): step size for central difference method
        '''

        batch_size, n_points, _ = points.shape
        device = self._device

        # if (not (type(c) == dict)) and c.shape[-1] != 0:
        #     c = c.unsqueeze(1).repeat(1, 6, 1).view(-1, c.shape[-1])
        # if (not (type(z) == dict)) and z.shape[-1] != 0:
        #     z = z.unsqueeze(1).repeat(1, 6, 1).view(-1, z.shape[-1])

        # calculate steps x + h/2 and x - h/2 for all 3 dimensions
        step = torch.cat([
            torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
        ], dim=1).to(device).unsqueeze(0) * h / 2



        points_eval = (points.unsqueeze(2).repeat(1, 1, 6, 1) + step).view(batch_size, n_points * 6, 3)

        # Eval decoder at these points
        #f = self.decoder(points_eval, z, c, only_occupancy=True, batchwise=False).view(batch_size, n_points, 6)
        f = self.decoder(points_eval, z, c, only_occupancy=True).view(batch_size, n_points, 6)

        # Get approximate derivate as f(x + h/2) - f(x - h/2)
        df_dx = torch.stack([
            (f[:, :, 0] - f[:, :, 1]),
            (f[:, :, 2] - f[:, :, 3]),
            (f[:, :, 4] - f[:, :, 5]),
        ], dim=-1)
        return df_dx

    def decode(self, p, z=None, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, z, c, only_occupancy=True, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def march_along_ray(self, ray0, ray_direction, z=None, c=None, it=None,
                        sampling_accuracy=None):
        ''' Marches along the ray and returns the d_i values in the formula
            r(d_i) = ray0 + ray_direction * d_i
        which return the surfaces points.

        Here, ray0 and ray_direction are directly used without any
        transformation; Hence the evaluation is done in object-centric
        coordinates.

        Args:
            ray0 (tensor): ray start points (camera centers)
            ray_direction (tensor): direction of rays; these should be the
                vectors pointing towards the pixels
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        device = self._device

        d_i = self.call_depth_function(ray0, ray_direction, self.decoder, z=z,
                                       c=c, it=it, n_steps=sampling_accuracy)

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        d_hat = torch.ones_like(d_i).to(device)
        d_hat[mask_pred] = d_i[mask_pred]
        d_hat[mask_zero_occupied] = 0.

        return d_hat, mask_pred, mask_zero_occupied

    def pixels_to_world(self, pixels, camera_mat, world_mat, scale_mat, z, c,
                        it=None, sampling_accuracy=None):
        ''' Projects pixels to the world coordinate system.

        Args:
            pixels (tensor): sampled pixels in range [-1, 1]
            camera_mat (tensor): camera matrices
            world_mat (tensor): world matrices
            scale_mat (tensor): scale matrices
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        batch_size, n_points, _ = pixels.shape
        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
                                             scale_mat)
        camera_world = origin_to_world(n_points, camera_mat, world_mat,
                                       scale_mat)
        ray_vector = (pixels_world - camera_world)

        d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(
            camera_world, ray_vector, z, c, it, sampling_accuracy)
        p_world_hat = camera_world + ray_vector * d_hat.unsqueeze(-1)
        return p_world_hat, mask_pred, mask_zero_occupied

    def decode_color(self, p_world, z=None, c=None, **kwargs):
        ''' Decodes the color values for world points.

        Args:
            p_world (tensor): world point tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        if hasattr(self, 'texture_field'):
            rgb_hat = self.texture_field(p_world, z=z, c=c, **kwargs)
        else:
            rgb_hat = self.decoder(p_world, z=z, c=c, only_texture=True)

        rgb_hat = torch.sigmoid(rgb_hat)
        return rgb_hat

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0).to(self._device)

        return c

    def infer_z(self, inputs, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(inputs, c, **kwargs)
        else:
            batch_size = inputs.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
