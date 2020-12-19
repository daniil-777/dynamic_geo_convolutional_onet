import os
import trimesh
import numpy as np 
import random 
from tqdm import tqdm
import sys
sys.path.append('.')
from im2mesh.utils.libmesh import check_mesh_contains
from copy import deepcopy

# fix random seed
np.random.seed(0)
random.seed(0)

input_path_watertight = '/is/rg/avg/lmescheder/project_data/2018/mesh_funcspace/data/OccProc/PointData.build'
# folder structure: classes -> 2_watertight -> *.off
input_path_splits = '/is/rg/avg/lmescheder/project_data/2018/mesh_funcspace/data/OccProc/PointDataChoy2016'
# folder structure: classes -> {train, val, test}.lst
output_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/LargeRoomDataset'
output_path_scene = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/scenes_large'
# classes to use from ShapeNet
classes = ['04256520', '03636649', '03001627', '04379243', '02933112']
classes = ['03001627']
classes.sort()

# number of rooms for train / val / test set
dataset_size = 10
split_percentages = [.75, .05, .2]
split_percentages = [1.]
n_rooms = [int(p * dataset_size) for p in split_percentages]
split_names = ['train', 'val', 'test']
split_names = ['train']
# dataset specfific hyperparameters
n_objects = [1, ]  # these will be considered individual classes!
n_repeat_object = 16
n_repeat_object = 64
scale_interval = [0.15, 0.15]
n_pointcloud_points = 100000
n_pointcloud_files = 10
n_iou_poins = 100000
n_iou_points_files = 10
use_groundplane = True
groundplane_thickness = 0.01
packbits = True
points_dtype = np.float32
save_scene_files = False


def create_dir(in_dir):
    ''' Creates directory if it does not exist
    '''
    if not os.path.isdir(in_dir):
        os.makedirs(in_dir)


def get_class_models(cl, split_name):
    ''' Returns list of models for specific class and split.
    '''
    split_file = os.path.join(input_path_splits, cl, '%s.lst' % split_name)
    with open(split_file, 'r') as f:
        model_files = f.read().split('\n')
        model_files = list(filter(lambda x: len(x) > 0, model_files))
    return model_files


def sample_models(model_dict, n_objects):
    ''' Samples n_objects from model_dict
    '''
    classes = [cl for cl in model_dict.keys()]
    classes.sort()

    out_list = []
    cl_list = []
    for n_object in range(n_objects):
        cl = random.choice(classes)
        cl_list.append(cl)
        model = random.choice(model_dict[cl])
        out_list.append('%s/%s' % (cl, model))
    return out_list, cl_list


def load_meshes(model_list, scale_list, n_repeat_object):
    ''' Loads the meshes in the list and scales according to provided list
    '''
    out_list = []
    for model_idx, model in enumerate(model_list):
        cl, m = model.split('/')
        model_path = os.path.join(input_path_watertight, cl, '2_watertight', '%s.off' % m)
        mesh = trimesh.load(model_path, process=False)
        # Apply scaling
        bbox = mesh.bounds
        current_scale = (bbox[1] - bbox[0]).max()
        mesh.vertices = mesh.vertices / current_scale * scale_list[model_idx]
        # Orientate mesh on z = -0.5 or on ground plane
        z_level = -0.5 + groundplane_thickness if use_groundplane else -0.5
        mesh.vertices[:, 1] = mesh.vertices[:, 1] - min(mesh.vertices[:, 1]) + z_level
        for repeat_idx in range(n_repeat_object):
            out_list.append(deepcopy(mesh))
    return out_list


def draw_sample(bounds, it=0, method='uniform', dx=0.1, sigma=0.05):
    ''' Draws a sample for provided method and given bounding box.
    '''
    if method == 'uniform':
        loc = -0.5 + np.random.rand(2) * (1 - bounds)
    if method == 'gaussian':
        mu_list = [[-0.5 + dx, -0.5 + dx], 
              [0.5 - dx, -0.5 + dx],
              [-0.5 + dx, 0.5 - dx],
              [0.5 - dx, 0.5 - dx],
              [0., 0.],
             ]
        while(True):
            loc = mu_list[it] + np.random.randn(2) * sigma
            if np.all(loc > -0.5) and np.all(loc + bounds < 0.5):
                break
    if method == 'uniform_structured':
        loc0 = [
            [-0.5, -0.5],
            [-0.5, 0.],
            [0., -0.5],
            [0., 0.],
        ]
        loc = loc0[it] + np.random.rand(2) * (0.5 - bounds)

    return loc

    

def check_intersection_interval(i1, i2):
    ''' Checks if the 2D intervals intersect.
    '''
    # i1, i2 of shape 2 x 2
    center_i1 = np.sum(i1, axis=0) / 2.
    center_i2 = np.sum(i2, axis=0) / 2.
    width_i1 = i1[1] - i1[0]
    width_i2 = i2[1] - i2[0]

    return np.all(abs(center_i1 - center_i2) < (width_i1 + width_i2) / 2)


def sample_locations(mesh_list, n_repeat_object=64, max_iter=1000):
    ''' Samples locations for the provided mesh list.
    '''
    bboxes = []

    bounds = (mesh_list[0].bounds[1] - mesh_list[0].bounds[0])[[0, 2]]
    n_row = int(np.sqrt(n_repeat_object))
    step_size = (1. - bounds) / (n_row - 1)
    x, y = np.meshgrid(np.arange(n_row), np.arange(n_row))
    xy = np.stack([x, y], axis=-1).reshape(-1, 2)

    loc0_array = np.array([-0.5, -0.5]).reshape(1, 2) + xy * step_size
    if np.any(loc0_array[-1] + bounds > 0.5):
        raise ValueError("Model is too large!")

    for mesh_idx, mesh in enumerate(mesh_list):
        # get bounds
        # bounds = (mesh.bounds[1] - mesh.bounds[0])[[0, 2]]
        # sample location
        # found_loc = False
        # it = 0
        # while(not found_loc):
        #     it += 1
        #     if it > max_iter:
        #         raise ValueError("Maximum number of iterations exceeded!")
        #     #loc0 = -0.5 + np.random.rand(2) * (1 - bounds)
        #     loc0 = draw_sample(bounds, method='uniform', it=mesh_idx)
        #     bbox_i = np.array([loc0, loc0 + bounds])
        #     found_loc = True
        #     for bbox in bboxes:
        #         if check_intersection_interval(bbox_i, bbox):
        #             found_loc = False
        #             break
        loc0 = loc0_array[mesh_idx]
        bbox_i = np.array([loc0, loc0 + bounds])
        bboxes.append(bbox_i)
        # scale mesh
        mesh.vertices[:, [0, 2]] = mesh.vertices[:, [0, 2]] - np.min(mesh.vertices[:, [0, 2]], axis=0).reshape(1, 2) + loc0.reshape(1, 2)
    return bboxes



def sample_scales(n_objects):
    '''Samples n_objects scales in intervl scale_interval
    '''
    out_list = [scale_interval[0] + np.random.rand() * (scale_interval[1] - scale_interval[0]) for i in range(n_objects)]
    return out_list


def sample_pointcloud(mesh_list, cl_list, ground_plane_only_above=True):
    ''' Samples point cloud from mesh list. 

    mesh_list (list): list of meshes. We assume that -1 is the ground plane
    '''
    c_vol = np.array([mesh.area for mesh in mesh_list])
    # if ground_plane_only_above:
    #     # roughly area is reduced by half if we only consider surface above
    #     c_vol[-1] = c_vol[-1] / 2
    c_vol /= sum(c_vol) 
    n_points = [int(c * n_pointcloud_points) for c in c_vol]
    points, normals, semantics = [], [], []
    for i, mesh in enumerate(mesh_list):
        # if i == len(mesh_list) - 1:
        #     cl_idx = -1
        # else:
        #     cl = cl_list[i]
        #     cl_idx = classes.index(cl)

        pi, face_idx = mesh.sample(n_points[i], return_index=True)
        if (i == len(mesh_list) - 1) and ground_plane_only_above:
            mask = pi[:, 1] > -0.5 + groundplane_thickness - 1e-4
            pi, face_idx = pi[mask], face_idx[mask]
        normals_i = mesh.face_normals[face_idx]
        # semantics_i = np.full_like(pi[:, 0], cl_idx)
        points.append(pi)
        normals.append(normals_i)
        # semantics.append(semantics_i)
        

    out_dict = {
        'points': np.concatenate(points, axis=0).astype(points_dtype),
        'normals': np.concatenate(normals, axis=0).astype(points_dtype),
        # 'semantics': np.concatenate(semantics, axis=0).astype(int)
    }

    return out_dict


def sample_iou_points(mesh_list, cl_list, iou_dir, scale_to_cube=True, padding=0.1, scale_by_constant=False, illustrate=False):
    y_max = max([max(mesh.vertices[:, 1]) for mesh in mesh_list])
    if scale_to_cube:
        scale_factor = 1 / (y_max - (-0.5 + groundplane_thickness))
    elif scale_by_constant:
        scale_factor = 1 / (scale_interval[1] - (-0.5 + groundplane_thickness))
    else:
        scale_factor = 0
    points = (np.random.rand(n_iou_poins * n_iou_points_files, 3).astype(np.float32) - 0.5) * (1 + padding)
    occ = np.zeros(n_iou_poins * n_iou_points_files).astype(bool)
    # semantics = np.ones(n_iou_poins * n_iou_points_files).astype(int) * len(mesh_list)
    for mesh_idx, mesh in enumerate(mesh_list):
        if mesh_idx != len(mesh_list) - 1 and (scale_to_cube or scale_by_constant):
            mesh.vertices[:, 1] = ((mesh.vertices[:, 1] + (0.5 - groundplane_thickness)) * scale_factor) - (0.5 + groundplane_thickness)
        occi = check_mesh_contains(mesh, points)
        occ = occ | occi
        # if mesh_idx == len(mesh_list) - 1:
        #     cl_idx = -1
        # else:
        #     cl_idx = classes.index(cl_list[mesh_idx])
        # semantics[occi == 1] = cl_idx
    
    # reshape for indivial files
    points = points.reshape(n_iou_points_files, n_iou_poins, 3)
    occ = occ.reshape(n_iou_points_files, n_iou_poins)
    # semantics = semantics.reshape(n_iou_points_files, n_iou_poins)


    points = points.astype(points_dtype)
    # semantics = semantics.astype(int)

    for file_idx in range(n_iou_points_files):
        out_dict = {
            'points': points[file_idx],
            'occupancies': np.packbits(occ[file_idx]),
            'z_scale': scale_factor,
            # 'semantics': semantics[file_idx],
        }
        np.savez(os.path.join(iou_dir, 'points_iou_%02d.npz' % file_idx), **out_dict)

    if illustrate:
        v = out_dict['points'][np.unpackbits(out_dict['occupancies']) == 1]
        pcl = trimesh.Trimesh(vertices=v, process=False)
        pcl.export(os.path.join(iou_dir, 'points.ply'))


def get_ground_plane():
    ground_plane = trimesh.creation.box((1., groundplane_thickness, 1.))
    ground_plane.vertices[:, 1] = ground_plane.vertices[:, 1] - min(ground_plane.vertices[:, 1]) - 0.5
    return ground_plane


def merge_meshes(mesh_list):
    out_mesh = mesh_list[0]
    for mesh in mesh_list[1:]:
        n_vertices = out_mesh.vertices.shape[0]
        v = mesh.vertices
        f = mesh.faces + n_vertices
        n = mesh.face_normals
        out_mesh = trimesh.Trimesh(
            vertices = np.concatenate([out_mesh.vertices, v]),
            faces = np.concatenate([out_mesh.faces, f]),
            face_normals=np.concatenate([out_mesh.face_normals, n]),
            process=False
        )
    return out_mesh


create_dir(output_path)
create_dir(output_path_scene)
# Main loop of data generation
pbar = tqdm(total=sum(n_rooms))
for n_object in n_objects:
    dataset_item = 0
    cl_path = os.path.join(output_path, 'rooms_%02d_%03d' % (n_object, n_repeat_object))
    create_dir(cl_path)
    for split_idx, split_name in enumerate(split_names):
        split_lst = ''
        model_files = {}
        for cl in classes:
            model_files[cl] = get_class_models(cl, split_name)
        # Loop over items for current split
        split_item_idx = 0
        while split_item_idx < n_rooms[split_idx]:
            # Create item folder
            item_dict = {}
            item_dict['room_idx'] = dataset_item
            item_dict['split'] = split_name
            item_dict['n_objects'] = n_object #np.random.randint(n_objects[0], n_objects[1])
            item_dict['n_repeat_object'] = n_repeat_object
            obj_list, cl_list = sample_models(model_files, n_object)
            item_dict['objects'] = obj_list
            item_dict['classes'] = cl_list
            item_dict['scales'] = sample_scales(n_object)
            meshes = load_meshes(item_dict['objects'], item_dict['scales'], item_dict['n_repeat_object'])
            try:
                bboxes = sample_locations(meshes, n_repeat_object=n_repeat_object)
                item_dict['bboxes'] = bboxes
                if use_groundplane:
                    meshes += [get_ground_plane()]

                folder_path = os.path.join(cl_path, '%08d' % dataset_item)
                create_dir(folder_path)

                np.savez(os.path.join(folder_path, 'item_dict.npz'), **item_dict)

                pointcloud_dir = os.path.join(folder_path, 'pointcloud')
                create_dir(pointcloud_dir)
                for pcl_idx in range(n_pointcloud_files):
                    pcl_dict = sample_pointcloud(meshes, cl_list)
                    np.savez(os.path.join(pointcloud_dir, 'pointcloud_%02d.npz' % pcl_idx), **pcl_dict)
                    if pcl_idx == 0:
                        trimesh.Trimesh(vertices=pcl_dict['points'], process=False).export(os.path.join(
                            folder_path, 'pointcloud0.ply'
                        ))
                
                iou_dir = os.path.join(folder_path, 'points_iou')
                create_dir(iou_dir)
                sample_iou_points(meshes, cl_list, iou_dir, scale_to_cube=False, scale_by_constant=False)

                if save_scene_files:
                    merged_mesh = merge_meshes(meshes)
                    merged_mesh.export(os.path.join(output_path_scene, '%08d.obj' % dataset_item))

                dataset_item += 1
                split_item_idx += 1

                if split_lst != '':
                    split_lst += '\n'
                split_lst += '%08d' % dataset_item
                pbar.update(1)

            except Exception as e:
                print('Error: ', e)
        pbar.close()
        with open(os.path.join(cl_path, split_name + '.lst'), 'w') as f:
            f.write(split_lst)
