import trimesh
import numpy as np
from os.path import join, exists, isdir
from os import listdir, makedirs
from tqdm import tqdm


# scene_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Matterport3D/2t7WUuJeko7'
# out_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/output/Matterport3D_fused'
# cut_mesh = True
#
# def create_dir(dir_in):
#     if not exists(dir_in):
#         makedirs(dir_in)
#
# create_dir(out_path)
# regions = [m for m in listdir(scene_path) if isdir(join(scene_path, m))]
#
#
# for idx, region in tqdm(enumerate(regions)):
#     r_path = join(scene_path, region)
#     mesh = trimesh.load(join(r_path, 'mesh_transformed.ply'))
#     transform = np.load(join(r_path, 'pointcloud.npz'))['transform']
#     transform = np.linalg.inv(transform)
#     mesh.apply_transform(transform)
#
#     # TODO optionally we can cut the room itself
#
#     if idx == 0:
#         faces = mesh.faces
#         vertices = mesh.vertices
#     else:
#         faces = np.concatenate([faces, mesh.faces + vertices.shape[0]])
#         vertices = np.concatenate([vertices, mesh.vertices])
#
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
# x_max = max(mesh.vertices[:, 2])
# # TODO here we cut it at the end
# if cut_mesh:
#     mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 0, -1]), np.array([0, 0, x_max - 0.1]))
#
# out_file = join(out_path, 'mesh_fused.ply')
# mesh.export(out_file)


import trimesh
import numpy as np
from os.path import join, exists, isdir
from os import listdir, makedirs
from tqdm import tqdm

# id = '1LXtFkjw3qL'
id_list = ['2azQ1b91cZZ', '2n8kARJN3HM', '2t7WUuJeko7', '5LpN3gDmAk7', '5q7pvUzZiYa', '5ZKStnWn8Zo', '8WUmhLawc2A',
           '17DRP5sb8fy', '29hnd4uzFmX', '82sE5b5pLXE', '759xd9YjKW5', '8194nk5LbLH', 'ARNzJeq3xxb', 'D7G3Y4RVNrH',
           'D7N2EKCX4Sj', 'E9uDoFAP3SH', 'EDJbREhghzL', 'EU6Fwq7SyZv', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'JeFG25nYj2p',
           'JF19kD82Mey', 'JmbYfDe2QKZ', 'PuKPg4mmafe', 'PX4nDJXEHrG', 'QUCTc6BB5sX', 'RPmz2sHmrrY', 'S9hNv5qa7GM',
           'SN83YJsR3w2', 'TbHJrupSAjP', 'ULsKaCPVFJR', 'UwV83HsGsw3', 'Uxmj2M2itWa', 'V2XKFyX4ASd']

for id in id_list:
    scene_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Matterport3D/'+ id
    out_path = '../../out/matterport/grid64/generation_march4/meshes_fused/' + id
    in_path = '../../out/matterport/grid64/generation_march4/meshes/' + id

    cut_mesh = True

    def create_dir(dir_in):
        if not exists(dir_in):
            makedirs(dir_in)

    create_dir(out_path)
    regions = [m for m in listdir(scene_path) if isdir(join(scene_path, m))]


    for idx, region in tqdm(enumerate(regions)):
        r_path = join(scene_path, region)
        obj = r_path.split('/')[-1]
        mesh = trimesh.load(join(in_path, obj+'.off'))
        transform = np.load(join(r_path, 'pointcloud.npz'))['transform']
        transform = np.linalg.inv(transform)
        mesh.apply_transform(transform)

        z_max = max(mesh.vertices[:, 2])
        z_range = max(mesh.vertices[:, 2]) - min(mesh.vertices[:, 2])
        x_min = min(mesh.vertices[:, 0])
        y_min = min(mesh.vertices[:, 1])
        if cut_mesh:
            mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 0, -1]), np.array([0, 0, z_max - 0.5*z_range]))
            mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 1, 0]), np.array([0, y_min + 0.5, 0]))
            print(z_range)

        out_file = join(out_path, 'mesh_fused%d.ply'%idx)
        mesh.export(out_file)


        if idx == 0:
            faces = mesh.faces
            vertices = mesh.vertices
        else:
            faces = np.concatenate([faces, mesh.faces + vertices.shape[0]])
            vertices = np.concatenate([vertices, mesh.vertices])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    x_max = max(mesh.vertices[:, 2])
    # if cut_mesh:
    #     mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 0, -1]), np.array([0, 0, x_max - 1.5]))

    out_file = join(out_path, 'mesh_fused.ply')
    mesh.export(out_file)
