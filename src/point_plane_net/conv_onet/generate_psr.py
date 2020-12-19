import torch
import numpy as np
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid
from pypoisson import poisson_reconstruction
import trimesh
import tempfile
import open3d as o3d
import subprocess

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Count how many models already created
model_counter = defaultdict(int)
for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}


    points = data['inputs'][0]
    normals = data['inputs.normals'][0]

    if cfg['data']['psr_add_noise_to_normals']:
        std_dev = cfg['data']['pointcloud_noise']
        normals = normals + std_dev * torch.from_numpy(np.random.randn(*normals.shape)).float()

    if cfg['data']['psr_method'] == 'trim':
        trim_factor = cfg['data']['psr_trim_factor']
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = os.path.join(tmpdir, 'input_pointcloud.ply')
            pcd = o3d.PointCloud()
            pcd.points = o3d.Vector3dVector(points)
            pcd.normals = o3d.Vector3dVector(normals)
            o3d.write_point_cloud(tmp_file, pcd)
            tmp_mesh = os.path.join(tmpdir, 'mesh.ply')
            dirpath = os.getcwd()
            process = subprocess.Popen("bash call_poisson_mesher.sh %s %s" % (tmp_file, tmp_mesh), cwd=dirpath, shell=True).wait()
            if trim_factor == 0:
                mesh = trimesh.load(tmp_mesh, process=False)
            else:
                tmp_mesh_trim = os.path.join(tmpdir, 'mesh_trimmed.ply')
                process = subprocess.Popen("bash call_poisson_trimmer.sh %s %s %d" % (tmp_mesh, tmp_mesh_trim, trim_factor), cwd=dirpath, shell=True).wait()
                mesh = trimesh.load(tmp_mesh_trim, process=False)
    else:
        t00 = time.time()
        f_out, v_out = poisson_reconstruction(points, normals)
        mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)

    # Write output
    mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
    mesh.export(mesh_out_file)
    out_file_dict['mesh'] = mesh_out_file

    
    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
