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
eps = 0.001
# n_pts_pcl = 80000
# n_pts_uni = 20000
n_pixels = 100
n_images = 500

scenes = listdir(ds_path)
scenes.sort()

for scene in tqdm(scenes):
    scene_path = join(ds_path, scene)

    out_file = join(scene_path, 'points.npz')
    if os.path.exists(out_file):
        continue

    # these scenes fails?
    if scene == 'scene0514_00':
        continue

    cam = np.load(join(scene_path, 'cameras.npz'))
    mask_idx = cam.get('camera_mask', [])
    pcl = np.load(join(scene_path, 'pointcloud.npz'))['points'].astype(np.float32)
    bmin, bmax = torch.from_numpy(np.min(pcl, axis=0)), torch.from_numpy(np.max(pcl, axis=0))
    depth_files = glob.glob(join(scene_path, 'depth', '*.png'))
    depth_files.sort()

    p_out = []
    p_out2 = []

    idx_list = [i for i in range(len(depth_files)) if i not in mask_idx]
    idx_subsample = np.random.choice(idx_list, size=(min(len(idx_list), n_images),), replace=False)



    # for idx, depth_file in tqdm(enumerate(depth_files)):
    for idx in tqdm(idx_subsample):
        depth_file = depth_files[idx]
        cm = torch.from_numpy(cam['camera_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        wm = torch.from_numpy(cam['world_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)
        sm = torch.from_numpy(cam['scale_mat_%d' % idx].astype(np.float32)).view(1, 4, 4)

        depth = torch.from_numpy(imageio.imread(depth_file).astype(np.float32)) / 1000
        h, w = depth.shape
        depth = depth.view(1, 1, h, w)
        #pixels = arange_pixels(resolution=(h, w))[1]
        pixels = torch.rand(1, n_pixels, 2) * 2 - 1.
        d = get_tensor_values(depth, pixels)
        mask_gt = d[:,:,0] != 0
        p_world = transform_to_world(pixels, d+eps, cm, wm, sm)
        d_free = (0.25 + torch.rand_like(d) * 0.75) * d
        p_world_free = transform_to_world(pixels, d_free, cm, wm, sm)
        p_out.append(p_world[mask_gt])
        p_out2.append(p_world_free[mask_gt])
   
    
    # additionally check if point inside scan
    p_inside = torch.cat(p_out)
    mask = torch.all(((p_inside <= 0.5) &
                      (p_inside >= -0.5)), dim=-1)

    p_outside = torch.cat(p_out2)
    p_out = torch.cat([p_inside, p_outside])
    # p_out = torch.cat(p_out + p_out2)
    n_p = p_inside.shape[0]

    p_uni = (torch.rand(n_p, 3) - 0.5) * 1.1
    mask_outside = torch.any((p_uni < bmin.reshape(1, 3)) | (p_uni > bmax.reshape(1, 3)), dim=-1)
     
    #import ipdb; ipdb.set_trace()
    p_out = torch.cat([p_out, p_uni[mask_outside]]).numpy()
    occ = torch.zeros(p_out.shape[0])
    occ[:n_p] = 1.
    occ = occ.bool().numpy()
    out_file2 = join(scene_path, 'points_iou.ply')
    trimesh.Trimesh(vertices=p_out[:n_p], process=False).export(out_file2)
    out_dict = {
        'points': p_out.astype(np.float16),
        'occupancies': np.packbits(occ),
    }
    np.savez(out_file, **out_dict)
