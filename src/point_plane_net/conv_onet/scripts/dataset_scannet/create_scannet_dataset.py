import os
import time
import numpy as np
import argparse
import pdb
import trimesh
import occ_fusion
from SensorData import SensorData
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

directory = '/media/hdd/ScanNet/data_v2'
out_path = './scannet_processed'

if not os.path.exists(out_path):
	os.mkdir(out_path)

parser = argparse.ArgumentParser('Sample points from ScanNet dataset.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')


sampling_type = 'uniform'
points_dtype = np.float16
num_points = 100000
frame_skip = 1
padding = 0.0
truncation = 0.2
swap_yz = False


def align_axis(file_name, mesh):
	lines = open(file_name).readlines()
	for line in lines:
		if 'axisAlignment' in line:
			axis_align_matrix = [float(x) \
					for x in line.rstrip().strip('axisAlignment = ').split(' ')]
			break
	axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
	pts = np.ones((mesh.vertices.shape[0], 4))
	pts[:,0:3] = mesh.vertices[:,0:3]
	pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
	mesh.vertices = pts[:, 0:3]

	return mesh, axis_align_matrix

def process(file):

	t0 = time.time()
	out_path_cur = os.path.join(out_path, file)
	if not os.path.exists(out_path_cur):
		os.mkdir(out_path_cur)

	# load mesh
	mesh = trimesh.load(os.path.join(directory, file, file+'_vh_clean.ply'))
	# align axis
	mesh, axis_align_matrix = align_axis(os.path.join(directory, file, file+'.txt'), mesh)
	vol_bnds = np.transpose(mesh.bounds, (1,0)).copy()

	file_cur = os.path.join(directory, file, file+'.sens')
	sd = SensorData(file_cur)
	depth_im, cam_pose, cam_intr = sd.extract_depth_images(frame_skip=frame_skip)
	# vol_bnds = occ_fusion.compute_frustum(depth_im, cam_intr, cam_pose)
	occ_vol = occ_fusion.OccupancyFusionUniform(vol_bnds, axis_align_matrix, num_points=num_points, trunc=truncation)
	for i in tqdm(range(len(depth_im))):
		occ_vol.fuse(depth_im[i], cam_intr, cam_pose[i])

	points, vol_center, bound_max, occupancies = occ_vol.normalize_pts(padding=padding, swap_yz=swap_yz)
	occ_vol.save_occ(os.path.join(out_path_cur, 'points.npz'), points_dtype=points_dtype)
	trimesh.Trimesh(vertices=points[occupancies], process=False).export(os.path.join(out_path_cur, 'occupied.ply'))
	# trimesh.Trimesh(vertices=points, process=False).export(os.path.join(out_path_cur, 'points.ply'))

	# translate and scale the point clouds to [-0.5, 0.5]
	pi, face_idx = mesh.sample(num_points, return_index=True)
	pi = (pi - vol_center) / bound_max
	# Orientate mesh on z = -0.5
	pi[:, 2] = (pi[:, 2] - pi[:, 2].min()) - 0.5

	pi = (1 + padding) * pi

	# swap y and z axis
	if swap_yz:
		pi_tmp = pi.copy()
		pi[:, 1] = pi_tmp[:, 2]
		pi[:, 2] = pi_tmp[:, 1]

	normals = mesh.face_normals[face_idx]
	# save point cloud
	out_dict = {
				'points': pi.astype(points_dtype),
				'normals': normals.astype(points_dtype)
                }
	np.savez(os.path.join(out_path_cur, 'pointcloud.npz'), **out_dict)
	trimesh.Trimesh(vertices=out_dict['points'], process=False).export(os.path.join(
                            out_path_cur, 'pointcloud.ply'
                        ))
	print('processing time for {}: {}'.format(file, time.time()-t0))

if __name__ == '__main__':
	t0 = time.time()
	args = parser.parse_args()
	file_list = os.listdir(directory)
	if args.n_proc != 0:
		with Pool(args.n_proc) as p:
			p.map(partial(process), file_list)
	else:
		for file in tqdm(file_list):
			process(file)
	print('Finished, total processing time: ', time.time() - t0)

