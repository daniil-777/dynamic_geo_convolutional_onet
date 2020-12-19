import numpy as np
import os
import imageio
from os.path import join
from os import listdir
import glob
import sys
sys.path.append('.')
from im2mesh.dvr_common import transform_to_camera_space, get_tensor_values, arange_pixels, transform_to_world
import torch
import trimesh
from tqdm import tqdm


ds_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNet/scenes'
eps = 0.01
n_pts_pcl = 80000
n_pts_uni = 20000
subsample_views = 0
n_img_points = 5000

# scenes = listdir(ds_path)
# scenes.sort()
# scenes = [scenes[0]]

#scenes = ['scene0370_00']
scenes = ['scene0134_02']

for scene in scenes:
    scene_path = join(ds_path, scene)
    cam = np.load(join(scene_path, 'cameras.npz'))
    pcl = np.load(join(scene_path, 'pointcloud.npz'))['points'].astype(np.float32)
    bmin, bmax = torch.from_numpy(np.min(pcl, axis=0)), torch.from_numpy(np.max(pcl, axis=0))
    depth_files = glob.glob(join(scene_path, 'depth', '*.png'))
    depth_files.sort()
    print('Number of views: %d' % (len(depth_files)))

    if subsample_views > 0:
        skip_ids = np.random.choice(len(depth_files), size=(len(depth_files) - subsample_views,), replace=False)
    else:
        skip_ids = []

    p_out = []
    p_out2 = []
    for idx, depth_file in tqdm(enumerate(depth_files)):
        if idx in skip_ids:
            continue
        cm = torch.from_numpy(cam['camera_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        wm = torch.from_numpy(cam['world_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        sm = torch.from_numpy(cam['scale_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        depth = torch.from_numpy(imageio.imread(depth_file).astype(np.float32)) / 1000
        h, w = depth.shape
        depth = depth.view(1, 1, h, w)
        #pixels = arange_pixels(resolution=(h, w))[1]
        pixels = torch.rand(1, n_img_points, 2) * 2 - 1.
        d = get_tensor_values(depth, pixels)
        mask_gt = d[:,:,0] != 0
        add_eps = torch.rand_like(d) * eps
        p_world = transform_to_world(pixels, d+add_eps, cm, wm, sm)
        d_free = (0.25 + torch.rand_like(d) * 0.75) * d
        p_world_free = transform_to_world(pixels, d_free, cm, wm, sm)
        p_out.append(p_world[mask_gt])
        p_out2.append(p_world_free[mask_gt])

    p_out = torch.cat(p_out).unsqueeze(0)
    occ = torch.ones(1, p_out.shape[1]).bool()
    for idx, depth_file in tqdm(enumerate(depth_files)):
        if idx in skip_ids:
            continue
        cm = torch.from_numpy(cam['camera_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        wm = torch.from_numpy(cam['world_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        sm = torch.from_numpy(cam['scale_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        depth = torch.from_numpy(imageio.imread(depth_file).astype(np.float32)) / 1000
        h, w = depth.shape
        depth = depth.view(1, 1, h, w)
        p_cam = transform_to_camera_space(p_out, cm, wm, sm)
        pixels = p_cam[:, :, :2] / p_cam[:, :, -1].unsqueeze(-1)
        mask_pixel = (
            (pixels[:, :, 0] >= -1) &
            (pixels[:, :, 1] >= -1) &
            (pixels[:, :, 0] <= 1) &
            (pixels[:, :, 1] <= 1)
        )
        
        d_gt = get_tensor_values(depth, pixels).squeeze(-1)
        mask_gt = d_gt > 0
        d_hat = p_cam[:, :, -1]
        mask_pred = d_hat > 0

        mask = mask_pixel & mask_gt & mask_pred
        occ[mask] &= (d_hat >= d_gt)[mask]
    #p_out = torch.cat(p_out + p_out2)
    
    p_out = p_out[occ == 1]
    p_out = torch.cat([p_out, torch.cat(p_out2)],)
    n_p = (occ == 1).sum()
    p_uni = (torch.rand(n_p, 3) - 0.5) * 1.1
    mask_outside = torch.any((p_uni < bmin.reshape(1, 3)) | (p_uni > bmax.reshape(1, 3)), dim=-1)
     
    p_out = torch.cat([p_out, p_uni[mask_outside]]).numpy()
    occ = torch.zeros(p_out.shape[0])
    occ[:n_p] = 1.
    occ = occ.bool().numpy()
    
    out_file = join(scene_path, 'points_adv.npz')
    out_file2 = join(scene_path, 'points2_adv.ply')
    trimesh.Trimesh(vertices=p_out[occ==1], process=False).export(out_file2)
    out_dict = {
        'points': p_out.astype(np.float16),
        'occupancies': np.packbits(occ),
    }
    np.savez(out_file, **out_dict)
