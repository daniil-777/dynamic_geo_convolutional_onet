import os
import glob
import random
from PIL import Image
import numpy as np
from im2mesh.data.core import Field
import imageio
import torch


class IndexField(Field):
    ''' Basic index field.'''

    def load(self, model_path, idx, category, **kwargs):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''

    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class PointsIou(Field):
    def __init__(self, file_name='points_iou.npz', n_points=100000):
        self.file_name = file_name
        self.n_points = n_points

    def load(self, model_path, idx, category, input_idx_img=None):
        points_dict = np.load(os.path.join(model_path, self.file_name))
        p = points_dict['points'].astype(np.float32)
        p = np.stack([p[:, 2], p[:, 0], p[:, 1]], axis=-1)  # fix orientation
        occ = points_dict['occupancy'].astype(np.float32)
        voxel = points_dict['voxel'].astype(np.float32).transpose(2, 0, 1)
        # add downsampling
        data = {
            None: p,
            'occ': occ,
            'voxel': voxel,
        }
        return data


class SparsePointCloud(Field):
    def __init__(self, folder_name='image', file_name='pcl.npz',
                 random_view=True, ignore_image_idx=[]):
        self.file_name = file_name
        self.image_folder = folder_name
        self.random_view = random_view
        # TODO is this really starting at 0?
        # self.img_skip_list = list(range(3, 8)) + \
        #     list(range(16, 22)) + list(range(36, 40))
        #self.img_skip_list = [l-1 for l in self.img_skip_list]
        self.ignore_image_idx = ignore_image_idx

    def load(self, model_path, idx, category, input_idx_img=None):

        # TODO work around for now
        files = os.listdir(os.path.join(model_path, self.image_folder))
        # if len(files) == 49:
        #     img_list = [i for i in range(49) if i not in self.img_skip_list]
        # else:
        #     img_list = [i for i in range(65) if i not in self.img_skip_list]

        img_list = [i for i in range(len(files)) if i not in self.ignore_image_idx]

        if input_idx_img is not None:
            idx_img = input_idx_img
        elif self.random_view:
            idx_img = np.random.randint(0, len(img_list)-1)
        else:
            idx_img = 0

        file_path = os.path.join(model_path, self.file_name)
        npz_file = np.load(file_path)
        p = npz_file['points']
        is_in_visual_hull = npz_file['is_in_visual_hull']
        c = npz_file['colors']
        v = npz_file['visibility_%04d' % img_list[idx_img]]
        # v = npz_file['visibility_%04d' % idx_img]

        p = p[v][is_in_visual_hull[v]]
        c = c[v][is_in_visual_hull[v]]

        # load cam with
        camera_dict = np.load(os.path.join(
            os.path.join(model_path, 'cameras.npz')
        ))

        Rt = camera_dict['world_mat_%d' % img_list[idx_img]].astype(np.float32)
        K = camera_dict['camera_mat_%d' % img_list[idx_img]].astype(np.float32)
        S = camera_dict.get('scale_mat_%d' % img_list[idx_img]).astype(np.float32)

        data = {}

        phom = np.concatenate([
            p, np.ones((p.shape[0], 1))
        ], axis=-1).transpose(1, 0)

        p_proj = K @ Rt @ phom
        d = p_proj[-2]
        p_proj = p_proj[:2] / p_proj[-2].reshape(1, -1)
        p_proj = p_proj.transpose(1, 0)

        p_world = np.linalg.inv(S) @ phom
        p_world = p_world[:3].transpose(1, 0)

        data['p_world'] = p_world.astype(np.float32)
        data['p_img'] = p_proj.astype(np.float32)
        data['d'] = d.astype(np.float32)
        data['colors'] = c.astype(np.float32)
        data['world_mat'] = Rt.astype(np.float32)
        data['scale_mat'] = S.astype(np.float32)
        data['camera_mat'] = K.astype(np.float32)

        return data


class ImagesFieldDVR(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''

    def __init__(self, folder_name, mask_folder_name='mask',
                 depth_folder_name='depth', transform=None,
                 extension='jpg', mask_extension='png',
                 depth_extension='exr',
                 with_camera=False, with_mask=True, with_depth=False,
                 random_view=True,   dataset_name='Shapes3D',
                 visual_hull_depth_folder='visual_hull_depth',
                 img_with_index=False, all_images=False,
                 n_views=0,
                 n_images=1, depth_from_visual_hull=False, 
                 ignore_image_idx=[], use_mask_from_depth=True, **kwargs):
        self.folder_name = folder_name
        self.mask_folder_name = mask_folder_name
        self.depth_folder_name = depth_folder_name

        self.transform = transform

        self.extension = extension
        self.mask_extension = mask_extension
        self.depth_extension = depth_extension

        self.random_view = random_view
        self.n_views = n_views

        self.with_camera = with_camera
        self.with_mask = with_mask
        self.with_depth = with_depth

        self.dataset_name = dataset_name
        self.img_with_index = img_with_index
        self.all_images = all_images
        self.n_images = n_images

        self.depth_from_visual_hull = depth_from_visual_hull
        self.visual_hull_depth_folder = visual_hull_depth_folder
        self.ignore_image_idx = ignore_image_idx
        self.use_mask_from_depth = use_mask_from_depth

    def load(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the field.

        Args:
            model_path (str): path to model
            idx (int): model id
            category (int): category id
            input_idx_img (int): image id which should be used (this
                overwrites any other id)
        '''
        if self.all_images:
            n_files = self.get_number_files(model_path)
            data = {}
            for input_idx_img in range(n_files):
                datai = self.load_field(model_path, idx, category,
                                        input_idx_img)
                data['img%d' % input_idx_img] = datai
            data['n_images'] = n_files
            return data
        else:
            return self.load_field(model_path, idx, category, input_idx_img)

    def get_number_files(self, model_path, ignore_filtering=False):
        ''' Returns how many views are present for the model.

        Args:
            model_path (str): path to model
        ''' 
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()

        if not ignore_filtering and len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(len(files)) if idx not in self.ignore_image_idx]

        if not ignore_filtering and self.n_views > 0:
            files = files[:self.n_views]
        return len(files)

    def load_file_from_folder(self, model_path, folder_name, extension, idx):
        folder = os.path.join(model_path, folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % extension))
        files.sort()

        if len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(len(files)) if idx not in self.ignore_image_idx]

        if self.n_views > 0:
            files = files[:self.n_views]
        return files[idx]

    def load_image(self, model_path, idx, data={}):
        filename = self.load_file_from_folder(model_path, self.folder_name,
                                              self.extension, idx)
        image = Image.open(filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        data[None] = image

    def load_camera(self, model_path, idx, data={}, n_files=-1):

        camera_file = os.path.join(model_path, 'cameras.npz')
        camera_dict = np.load(camera_file)

        if len(self.ignore_image_idx) > 0:
            n_files = self.get_number_files(model_path, ignore_filtering=True)
            idx_list = [i for i in range(n_files) if i not in self.ignore_image_idx]
            idx_list.sort()
            idx = idx_list[idx]

        camera_file = os.path.join(model_path, 'cameras.npz')
        camera_dict = np.load(camera_file)
        Rt = camera_dict['world_mat_%d' % idx].astype(np.float32)
        K = camera_dict['camera_mat_%d' % idx].astype(np.float32)
        S = camera_dict.get(
            'scale_mat_%d' % idx, np.eye(4)).astype(np.float32)
        data['world_mat'] = Rt
        data['camera_mat'] = K
        data['scale_mat'] = S

    def load_mask(self, model_path, idx, data={}):
        if self.use_mask_from_depth:
            mask = data['depth'] != np.inf
        else:
            filename = self.load_file_from_folder(
                model_path, self.mask_folder_name, self.mask_extension, idx)
            mask = np.array(Image.open(filename)).astype(np.bool)
            mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
        data['mask'] = mask.astype(np.float32)

    def load_depth(self, model_path, idx, data={}):
        filename = self.load_file_from_folder(
            model_path, self.depth_folder_name, self.depth_extension, idx)
        depth = np.array(imageio.imread(filename)).astype(np.float32)
        depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
        data['depth'] = depth

    def load_visual_hull_depth(self, model_path, idx, data={}):
        filename = self.load_file_from_folder(
            model_path, self.visual_hull_depth_folder, self.depth_extension,
            idx)
        depth = np.array(imageio.imread(filename)).astype(np.float32)
        depth = depth.reshape(
            depth.shape[0], depth.shape[1], -1)[:, :, 0]
        data['depth'] = depth

    def load_field(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''

        n_files = self.get_number_files(model_path)
        # For caching
        if input_idx_img is not None:
            idx_img = input_idx_img
        elif self.random_view:
            idx_img = random.randint(0, n_files - 1)
        else:
            idx_img = 0

        # Load the data
        data = {}
        self.load_image(model_path, idx_img, data)
        if self.with_camera:
            self.load_camera(model_path, idx_img, data, n_files)
        if self.with_depth:
            self.load_depth(model_path, idx_img, data)
        if self.with_mask:
            self.load_mask(model_path, idx_img, data)
        if self.depth_from_visual_hull:
            self.load_visual_hull_depth(model_path, idx_img, data)
        if self.img_with_index:
            data['idx'] = torch.tensor([idx_img]).float()
        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete

# class ImagesFieldDVR(Field):
#     ''' Image Field.

#     It is the field used for loading images.

#     Args:
#         folder_name (str): folder name
#         transform (list): list of transformations applied to loaded images
#         extension (str): image extension
#         random_view (bool): whether a random view should be used
#         with_camera (bool): whether camera data should be provided
#     '''

#     def __init__(self, folder_name, mask_folder_name='mask',
#                  depth_folder_name='depth', transform=None,
#                  extension='jpg', mask_extension='png',
#                  depth_extension='exr',
#                  with_camera=False, with_mask=True, with_depth=False,
#                  random_view=True,   dataset_name='Shapes3D',
#                  visual_hull_depth_folder='visual_hull_depth',
#                  img_with_index=False, all_images=False,
#                  n_views=0,
#                  n_images=1, depth_from_visual_hull=False, 
#                  ignore_image_idx=[], **kwargs):
#         self.folder_name = folder_name
#         self.mask_folder_name = mask_folder_name
#         self.depth_folder_name = depth_folder_name

#         self.transform = transform

#         self.extension = extension
#         self.mask_extension = mask_extension
#         self.depth_extension = depth_extension

#         self.random_view = random_view
#         self.n_views = n_views

#         self.with_camera = with_camera
#         self.with_mask = with_mask
#         self.with_depth = with_depth

#         self.dataset_name = dataset_name
#         self.img_with_index = img_with_index
#         self.all_images = all_images
#         self.n_images = n_images

#         self.depth_from_visual_hull = depth_from_visual_hull
#         self.visual_hull_depth_folder = visual_hull_depth_folder
#         self.ignore_image_idx = ignore_image_idx

#     def load(self, model_path, idx, category, input_idx_img=None):
#         ''' Loads the field.

#         Args:
#             model_path (str): path to model
#             idx (int): model id
#             category (int): category id
#             input_idx_img (int): image id which should be used (this
#                 overwrites any other id)
#         '''
#         if self.all_images:
#             n_files = self.get_number_files(model_path)
#             data = {}
#             for input_idx_img in range(n_files):
#                 datai = self.load_field(model_path, idx, category,
#                                         input_idx_img)
#                 data['img%d' % input_idx_img] = datai
#             data['n_images'] = n_files
#             return data
#         else:
#             return self.load_field(model_path, idx, category, input_idx_img)

#     def get_number_files(self, model_path, ignore_filtering=False):
#         ''' Returns how many views are present for the model.

#         Args:
#             model_path (str): path to model
#         ''' 
#         folder = os.path.join(model_path, self.folder_name)
#         files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
#         files.sort()

#         if not ignore_filtering and len(self.ignore_image_idx) > 0:
#             files = [files[idx] for idx in range(len(files)) if idx not in self.ignore_image_idx]

#         if not ignore_filtering and self.n_views > 0:
#             files = files[:self.n_views]
#         return len(files)

#     def load_file_from_folder(self, model_path, folder_name, extension, idx):
#         folder = os.path.join(model_path, folder_name)
#         files = glob.glob(os.path.join(folder, '*.%s' % extension))
#         files.sort()

#         if len(self.ignore_image_idx) > 0:
#             files = [files[idx] for idx in range(len(files)) if idx not in self.ignore_image_idx]

#         if self.n_views > 0:
#             files = files[:self.n_views]
#         return files[idx]

#     def load_image(self, model_path, idx, data={}):
#         filename = self.load_file_from_folder(model_path, self.folder_name,
#                                               self.extension, idx)
#         image = Image.open(filename).convert("RGB")
#         if self.transform is not None:
#             image = self.transform(image)
#         data[None] = image

#     def load_camera(self, model_path, idx, data={}, n_files=-1):

#         camera_file = os.path.join(model_path, 'cameras.npz')
#         camera_dict = np.load(camera_file)

#         if len(self.ignore_image_idx) > 0:
#             n_files = self.get_number_files(model_path, ignore_filtering=True)
#             idx_list = [i for i in range(n_files) if i not in self.ignore_image_idx]
#             idx_list.sort()
#             idx = idx_list[idx]

#         camera_file = os.path.join(model_path, 'cameras.npz')
#         camera_dict = np.load(camera_file)
#         Rt = camera_dict['world_mat_%d' % idx].astype(np.float32)
#         K = camera_dict['camera_mat_%d' % idx].astype(np.float32)
#         S = camera_dict.get(
#             'scale_mat_%d' % idx, np.eye(4)).astype(np.float32)
#         data['world_mat'] = Rt
#         data['camera_mat'] = K
#         data['scale_mat'] = S

#     def load_mask(self, model_path, idx, data={}):
#         filename = self.load_file_from_folder(
#             model_path, self.mask_folder_name, self.mask_extension, idx)
#         mask = np.array(Image.open(filename)).astype(np.bool)
#         mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
#         data['mask'] = mask.astype(np.float32)

#     def load_depth(self, model_path, idx, data={}):
#         filename = self.load_file_from_folder(
#             model_path, self.depth_folder_name, self.depth_extension, idx)
#         depth = np.array(imageio.imread(filename)).astype(np.float32)
#         depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
#         data['depth'] = depth

#     def load_visual_hull_depth(self, model_path, idx, data={}):
#         filename = self.load_file_from_folder(
#             model_path, self.visual_hull_depth_folder, self.depth_extension,
#             idx)
#         depth = np.array(imageio.imread(filename)).astype(np.float32)
#         depth = depth.reshape(
#             depth.shape[0], depth.shape[1], -1)[:, :, 0]
#         data['depth'] = depth

#     def load_field(self, model_path, idx, category, input_idx_img=None):
#         ''' Loads the data point.

#         Args:
#             model_path (str): path to model
#             idx (int): ID of data point
#             category (int): index of category
#         '''

#         n_files = self.get_number_files(model_path)
#         # For caching
#         if input_idx_img is not None:
#             idx_img = input_idx_img
#         elif self.random_view:
#             idx_img = random.randint(0, n_files - 1)
#         else:
#             idx_img = 0

#         # Load the data
#         data = {}
#         self.load_image(model_path, idx_img, data)
#         if self.with_camera:
#             self.load_camera(model_path, idx_img, data, n_files)
#         if self.with_mask:
#             self.load_mask(model_path, idx_img, data)
#         if self.with_depth:
#             self.load_depth(model_path, idx_img, data)
#         if self.depth_from_visual_hull:
#             self.load_visual_hull_depth(model_path, idx_img, data)
#         if self.img_with_index:
#             data['idx'] = torch.tensor([idx_img]).float()
#         return data

#     def check_complete(self, files):
#         ''' Check if field is complete.

#         Args:
#             files: files
#         '''
#         complete = (self.folder_name in files)
#         # TODO: check camera
#         return complete


class CameraField(Field):
    ''' Image Field.

    It is the field used for loading the camera dictionary.

    Args:
        n_views (int): number of views
        as_float (bool): whether to return the matrices as float
         (instead of double)
    '''

    def __init__(self, n_views, as_float=True):
        self.n_views = n_views
        self.as_float = as_float

    #def load(self, model_path, **kwargs):
    def load(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the camera field.

        Args:
            model_path (str): path to model
        '''
        camera_file = os.path.join(model_path, 'cameras.npz')
        cam = np.load(camera_file)
        data = {}
        dtype = np.float32 if self.as_float else np.float64
        for i in range(self.n_views):
            data['camera_mat_%d' % i] = cam.get(
                'camera_mat_%d' % i).astype(dtype)
            data['world_mat_%d' % i] = cam.get(
                'world_mat_%d' % i).astype(dtype)
            data['scale_mat_%d' % i] = cam.get(
                'scale_mat_%d' % i, np.eye(4)).astype(dtype)

        return data


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''

    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
