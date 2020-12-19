from os import listdir, makedirs
from os.path import join, exists
import json
import trimesh
import numpy as np 
from copy import deepcopy
import zipfile
from tqdm import tqdm


matterport_path = '/is/rg/avg/datasets/Matterport3D/v1/scans'

# scene_path_in = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Matterport_input/2t7WUuJeko7/region_segmentations'
# scene_name = '2t7WUuJeko7'
out_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Matterport3D'
out_path_data = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Matterport_input'
n_regions = 5
n_pointcloud_points = 100000
dtype = np.float16


def create_dir(dir_in):
    if not exists(dir_in):
        makedirs(dir_in)

create_dir(out_path)

def get_unit_cube_transform(mesh, permute = True):
    # Scale to unit cube
    bounds = pcd.bounds
    loc = bounds.sum(0) / 2.
    scale = max(bounds[1] - bounds[0])
    transform = np.eye(4).astype(np.float32)
    transform[:3, :3] /= scale
    transform[:-1, -1] = -loc / scale

    # Rotate mesh
    mat_permute = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    transform = mat_permute @ transform

    # Move y axis to -0.5
    mesh_transformed = deepcopy(mesh)
    mesh_transformed.apply_transform(transform)
    y_min = min(mesh_transformed.vertices[:, 1])

    mat_translate = np.eye(4)
    mat_translate[1, -1] = -0.5 - y_min

    transform = mat_translate @ transform

    return transform, loc, scale

# copy zip files
scan_ids = listdir(matterport_path)
scan_ids.sort()
for scan_id in tqdm(scan_ids):
    try:
        class_path = join(out_path, scan_id)
        if exists(class_path):
            continue
        else:
            create_dir(class_path)

        in_file = join(matterport_path, scan_id, 'region_segmentations.zip')
        with zipfile.ZipFile(in_file, 'r') as zip_ref:
            zip_ref.extractall(out_path_data)
        
        scene_path_in = join(out_path_data, scan_id, 'region_segmentations') 
        n_regions = int(len(listdir(scene_path_in)) / 4)
        for region in tqdm(range(n_regions)):
            region_path = join(class_path, 'region%03d' % region)
            create_dir(region_path)
            
            pcd = trimesh.load(join(scene_path_in, 'region%d.ply' % region), process=False)
            transform, loc, scale = get_unit_cube_transform(pcd)
            pcd.apply_transform(transform)

            pcd.export(join(region_path, 'mesh_transformed.ply'))

            pcl, face_idx = pcd.sample(n_pointcloud_points, return_index=True)
            normals = pcd.face_normals[face_idx]

            out_file = join(region_path, 'pointcloud.npz')
            np.savez(out_file, points=pcl.astype(dtype), normals=normals.astype(dtype), transform=transform, loc=loc, scale=scale)
    except Exception as e:
        print('Error for scene: %s' % scan_id)