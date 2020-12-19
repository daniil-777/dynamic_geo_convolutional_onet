import os
import numpy as np
import trimesh
from os.path import join
from os import listdir

ds_path = '/is/rg/avg/datasets/scannet'
ds_path_v2 = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/ScanNetv2/scans'
test_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data'

rotation_matrix = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
])

def align_axis(file_name, mesh):
        lines = open(file_name).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        axis_align_matrix = rotation_matrix @ axis_align_matrix
        mesh.apply_transform(axis_align_matrix)
        return mesh, axis_align_matrix


scene_ids = listdir(ds_path_v2)
scene_ids.sort()
scene_ids = [scene_ids[i] for i in [0, 14]]

for sid in scene_ids:
   mesh = trimesh.load(join(ds_path, sid, '%s_vh_clean.ply' % sid), process=False) 
   txt = join(ds_path_v2, sid, '%s.txt' % sid)
   mesh, mat = align_axis(txt, mesh)
   mesh.export(join(test_path, '%s.ply' % sid))
