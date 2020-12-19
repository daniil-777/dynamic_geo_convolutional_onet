import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.dvr import models, training, generation, rendering
from im2mesh import data


def get_model(cfg, device=None, len_dataset=0, **kwargs):
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
    depth_function_kwargs = cfg['model']['depth_function_kwargs']

    # Load always the decoder
    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )

    # Load encoder
    if encoder == 'idx':
        encoder = nn.Embedding(len_dataset, c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, **encoder_kwargs)
    else:
        encoder = None

    # Load latent encoder is z_dim not 0
    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            z_dim=z_dim, c_dim=c_dim, **encoder_latent_kwargs)
    else:
        encoder_latent = None

    # Initialize full model
    p0_z = get_prior_z(cfg, device)

    model = models.OccupancyNetwork(
        decoder, encoder=encoder, device=device,
        depth_function_kwargs=depth_function_kwargs,
        encoder_latent=encoder_latent, p0_z=p0_z,)
    return model


def get_trainer(model, optimizer, cfg, device, generator=None, **kwargs):
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
    n_training_points = cfg['data']['n_training_points']
    dataset_name = cfg['data']['dataset_name']
    lambda_freespace = 1. #cfg['model']['lambda_freespace']
    lambda_occupied = 1. #cfg['model']['lambda_occupied']
    lambda_rgb = 0. #cfg['model']['lambda_rgb']
    n_eval_points = 4000 #cfg['training']['n_eval_points']
    lambda_depth = 1. #cfg['model']['lambda_depth']
    lambda_image_gradients = 0. #cfg['model']['lambda_image_gradients']
    patch_size = 1. #cfg['model']['patch_size']
    reduction_method = 'sum' #cfg['model']['reduction_method']
    sample_continuous = False #cfg['training']['sample_continuous']
    lambda_sparse_depth = 0. #cfg['model']['lambda_sparse_depth']
    overwrite_visualization = False #cfg['training']['overwrite_visualization']
    depth_from_visual_hull = False #cfg['data']['depth_from_visual_hull']
    max_depth_value = 2.4 #cfg['data']['max_depth_value']
    depth_loss_on_world_points = False #cfg['training']['depth_loss_on_world_points']
    occupancy_random_normal = False #cfg['training']['occupancy_random_normal']
    use_cube_intersection = True #cfg['training']['use_cube_intersection']
    always_freespace = True #cfg['training']['always_freespace']
    multi_gpu = False #cfg['training']['multi_gpu']
    lambda_normal = 0.05 #cfg['model']['lambda_normal']
    lambda_iou = cfg['model']['lambda_iou']

    trainer = training.Trainer(
        model, optimizer, device=device, vis_dir=vis_dir, threshold=threshold,
        n_training_points=n_training_points, dataset_name=dataset_name,
        lambda_freespace=lambda_freespace, lambda_occupied=lambda_occupied,
        lambda_rgb=lambda_rgb, lambda_depth=lambda_depth, generator=generator,
        n_eval_points=n_eval_points,
        lambda_image_gradients=lambda_image_gradients,
        patch_size=patch_size, reduction_method=reduction_method,
        sample_continuous=sample_continuous,
        lambda_sparse_depth=lambda_sparse_depth,
        overwrite_visualization=overwrite_visualization,
        depth_from_visual_hull=depth_from_visual_hull,
        max_depth_value=max_depth_value,
        depth_loss_on_world_points=depth_loss_on_world_points,
        occupancy_random_normal=occupancy_random_normal,
        use_cube_intersection=use_cube_intersection,
        always_freespace=always_freespace, multi_gpu=multi_gpu,
        lambda_normal=lambda_normal, lambda_iou=lambda_iou,)

    return trainer


def get_renderer(model, cfg, device, **kwargs):
    ''' Returns the renderer object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    renderer = rendering.Renderer(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        colors=cfg['rendering']['colors'],
        resolution=cfg['rendering']['resolution'],
        n_views=cfg['rendering']['n_views'],
        extension=cfg['rendering']['extension'],
        background=cfg['rendering']['background'],
    )
    return renderer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        padding=cfg['generation']['padding'],
        with_color=cfg['generation']['with_colors'],
        refine_max_faces=cfg['generation']['refine_max_faces'],
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
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
    #resize_img_transform = data.ResizeImage(cfg['data']['img_size_train'])
    all_images = mode == 'render'
    with_depth = (cfg['model']['lambda_depth'] != 0)
    random_view = True if (
        mode == 'train' or
        ((cfg['data']['dataset_name'] == 'NMR') and mode == 'test') or
        ((cfg['data']['dataset_name'] == 'NMR') and mode == 'val')
    ) else False

    fields = {}
    if mode in ('train', 'val', 'render'):
        img_field = data.ImagesFieldDVR(
            cfg['data']['img_folder'], cfg['data']['mask_folder'],
            cfg['data']['depth_folder'],
            #transform=resize_img_transform,
            extension=cfg['data']['img_extension'],
            mask_extension=cfg['data']['mask_extension'],
            depth_extension=cfg['data']['depth_extension'],
            with_camera=True, #cfg['data']['img_with_camera'],
            with_mask=True,#cfg['data']['with_mask'],
            with_depth=with_depth,
            random_view=random_view,
            dataset_name=cfg['data']['dataset_name'],
            #img_with_index=cfg['data']['img_with_index'],
            all_images=all_images,
            n_views=cfg['data']['n_views'],
            #depth_from_visual_hull=cfg['data']['depth_from_visual_hull'],
            #visual_hull_depth_folder=cfg['data']['visual_hull_depth_folder'],
            #ignore_image_idx=cfg['data']['ignore_image_idx'],
        )
        fields['img'] = img_field

        # if cfg['model']['lambda_sparse_depth'] != 0:
        #     fields['sparse_depth'] = data.SparsePointCloud(
        #         ignore_image_idx=cfg['data']['ignore_image_idx'],
        #     )

        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
        # with_transforms = cfg['model']['use_camera']

        # fields['points'] = data.PointsField(
        #     cfg['data']['points_file'], points_transform,
        #     # with_transforms=with_transforms,
        #     unpackbits=cfg['data']['points_unpackbits'],
        #     # multi_files=cfg['data']['multi_files']
        # )

    elif cfg['data']['dataset_name'] == 'DTU':
        fields['camera'] = data.CameraField(
            cfg['data']['n_views'],
        )

    return fields
