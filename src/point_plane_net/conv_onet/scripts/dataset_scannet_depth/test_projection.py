import os
import numpy as np
import trimesh
import glob
import imageio
import sys
import torch
from tqdm import tqdm
sys.path.append('.')
from im2mesh.dvr_common import transform_to_world, arange_pixels, get_tensor_values

item_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNet_test/scenes'
item_path = os.path.join(item_path, os.listdir(item_path)[0])
n_views = 500

camera = np.load(os.path.join(item_path, 'cameras.npz'))
depth_files = glob.glob(os.path.join(item_path, 'depth', '*.png'))
depth_files.sort()

# import ipdb; ipdb.set_trace()
p = []
for i in tqdm(range(n_views)):
    depth_image = np.array(imageio.imread(depth_files[i]).astype(np.float32) / 1000.)
    #depth_image = depth_image.transpose(1, 0)
    h, w = depth_image.shape
    depth_image = depth_image.reshape(1, 1, h, w)
    _, pixels_dict = arange_pixels(resolution=(h, w))
    # pixels_dict = pixels_dict
    # xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # xx = xx.astype(np.float32) / (w - 1) * 2 - 1.
    # yy = yy.astype(np.float32) / (h - 1) * 2 - 1.
    # pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    #import ipdb; ipdb.set_trace()
    # depth_values = depth_image.reshape(-1, 1)

    depth_values_dict = get_tensor_values(depth_image.reshape(1, 1, h, w), pixels_dict, squeeze_channel_dim=True)
    depth_values_dict = depth_values_dict.reshape(depth_values_dict.shape[1], 1)

    # pixels_dict = pixels_dict[0].numpy()

    #mask = depth_values != 0
    camera_mat_dict = camera.get('camera_mat_%d' % i).reshape(1, 4, 4).astype(np.float32)
    # camera_mat = np.loadtxt('/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNet_depth8/scenes/scene0003_00/intrinsic/intrinsic_depth.txt').astype(np.float32)
    # camera_mat = np.array([
    #     [2./(h-1), 0, -1, 0],
    #     [0, 2./(w-1), -1, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ]).astype(np.float32) @ camera_mat
    
    world_mat_dict = camera.get('world_mat_%d' % i).reshape(1, 4, 4).astype(np.float32)
    #world_mat = np.linalg.inv(np.loadtxt('/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNet_depth8/scenes/scene0003_00/pose/%d.txt' % i)).astype(np.float32)
    # import ipdb; ipdb.set_trace()
    scale_mat = camera.get('scale_mat_%d' % i).reshape(1, 4, 4).astype(np.float32) 
    #scale_mat = np.eye(4).reshape(1, 4, 4).astype(np.float32)
    #pixels_hom = np.concatenate([pixels, np.ones((pixels.shape[0], 2))], axis=-1).transpose(1, 0)
    #pixels_hom[:3] *= depth_values.transpose(1, 0)
    #p_world =  np.linalg.inv(world_mat) @  np.linalg.inv(camera_mat) @ pixels_hom
    #p_world =  np.linalg.inv(world_mat_dict) @  np.linalg.inv(camera_mat_dict) @ pixels_hom
    #p_world = p_world[:3].transpose(1, 0)
    p_world = transform_to_world(pixels_dict, depth_values_dict.reshape(1, depth_values_dict.shape[0], 1), camera_mat_dict, world_mat_dict, scale_mat)
    #p_world = p_world[0]#[mask]
    p_world = p_world[0][depth_values_dict[:, 0] != 0]
    if torch.any(torch.isnan(p_world)):
        import ipdb; ipdb.set_trace()
    # subsample
    idx = np.random.choice(p_world.shape[0], size=(1000,), replace=False)
    p.append(p_world[idx])

p = np.concatenate(p)
trimesh.Trimesh(vertices=p, process=False).export('./out/test.ply')
