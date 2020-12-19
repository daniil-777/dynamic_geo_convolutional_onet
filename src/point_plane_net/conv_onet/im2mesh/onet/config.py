import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet import models, training, generation
from im2mesh import data
from im2mesh import config
from torchvision import transforms



import pdb


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    padding = cfg['data']['padding']

    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs
        )
    else:
        encoder_latent = None

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyNetwork(
        decoder, encoder, encoder_latent, p0_z, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    beta_vae = cfg['training']['beta_vae']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        beta_vae=beta_vae,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
        padding=cfg['data']['padding']
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    if ((cfg['model']['encoder_latent'] == 'pointnet_conv') or 
        (cfg['model']['encoder_latent'] == 'pointnet_conv2')):
        #TODO make nicer
        plane_res = cfg['model']['encoder_latent_kwargs'].get('plane_resolution', 128)
        n_conv_layer = cfg['model']['encoder_latent_kwargs'].get('n_conv_layer', 4)
        plane_type = cfg['model']['encoder_latent_kwargs'].get('plane_type', ['xz'])

        latent_dim = z_dim * (2 ** n_conv_layer)
        res_dim = int(plane_res / (2 ** n_conv_layer))
        
        if 'grid' in plane_type:
            spatial_resolution = (res_dim, ) * 3
        else:
            spatial_resolution = (res_dim, ) * 2

        p0_z = dist.Normal(
            torch.zeros((latent_dim, *spatial_resolution), device=device),
            torch.ones((latent_dim, *spatial_resolution), device=device)
        )
    else:
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    # TODO added this only as quick fix for Matterport3D
    if cfg['data']['path'].split('/')[-1] != 'Matterport3D':
        if not cfg['data']['points_online_sample']:
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        else:
            fields['points'] = data.OnlineSamplingPointsField(
                with_transforms=with_transforms,
                n_sample_points = cfg['data']['points_subsample'],
            )

    if cfg['data']['use_semantic_map']:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        fields['semantic_map'] = data.SemanticMapField(transform)

    # TODO added this only as quick fix for Matterport3D
    if mode in ('val', 'test') and cfg['data']['path'].split('/')[-1] != 'Matterport3D':
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
