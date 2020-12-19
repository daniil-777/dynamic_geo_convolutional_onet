import os
import numpy as np
from os import listdir, makedirs
from os.path import join, exists, isdir 
import trimesh 


ds_in = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/replica_v1'
ds_out = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/replica_processed'

def create_dir(in_dir):
    if not exists(in_dir):
        makedirs(in_dir)

def scale_mesh(mesh):
    bounds = mesh.bounds
    loc = (bounds.sum(0) / 2).reshape(1, 3)
    scale = (bounds[1] - bounds[0]).max()
    mesh.vertices = (mesh.vertices - loc) / scale
    return mesh

def rotate_mesh(mesh):
    R = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(R)
    return mesh


create_dir(ds_out)
ds_out = join(ds_out, 'scenes')
create_dir(ds_out)

scenes = listdir(ds_in)
scenes.sort()

scenes = [scenes[15]]

for scene in scenes:
    scene_path = join(ds_in, scene)
    scene_path_out = join(ds_out, scene)
    create_dir(scene_path_out)
    mesh_path = join(scene_path, 'mesh.ply')
    mesh = trimesh.load(mesh_path, process=False)
    mesh = scale_mesh(mesh)
    mesh = rotate_mesh(mesh)

    mesh_path_out = join(scene_path_out, 'mesh.ply')
    mesh.export(mesh_path_out)
