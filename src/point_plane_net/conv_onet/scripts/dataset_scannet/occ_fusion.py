# Codes are adapted from https://github.com/andyzeng/tsdf-fusion-python

import numpy as np
from skimage import measure
import pdb
import trimesh

def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
                               (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
                                np.array([0,max_depth,max_depth,max_depth,max_depth])])
    view_frust_pts = np.dot(cam_pose[:3,:3],view_frust_pts)+np.tile(cam_pose[:3,3].reshape(3,1),(1,view_frust_pts.shape[1])) # from camera to world coordinates
    return view_frust_pts

def compute_frustum(depth_im, cam_intr, cam_pose):
    vol_bnds = np.zeros((3,2))
    for i in range(len(depth_im)):
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im[i], cam_intr, cam_pose[i])
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0],np.amin(view_frust_pts,axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1],np.amax(view_frust_pts,axis=1))
    return vol_bnds

class OccupancyFusionUniform(object):

    def __init__(self, vol_bnds, axis_align_matrix, num_points=1000000, trunc=None):

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters

        # Adjust volume bounds
        self._vol_dim = self._vol_bnds[:, 1]-self._vol_bnds[:, 0]
        self._vol_bound_max = max(self._vol_dim)
        # Create a unit cube based on the maximum coordinates
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_bound_max
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)  # ensure C-order contigous
        
        # Pre-define 3D points within the valid volume
        self.world_pts = np.random.rand(3, num_points) * self._vol_bound_max + self._vol_origin.reshape(-1, 1)
        self.world_pts = self.world_pts.astype(np.float32)

        # transform the current world_pts back to the original axis
        axis_unalign_matrix = np.linalg.inv(axis_align_matrix)
        pts = np.ones((4, num_points))
        pts[:3, :] = self.world_pts
        self.world_pts_unalign = np.dot(axis_unalign_matrix, pts)[:3, :]

        self._occ_vol = np.zeros(num_points).astype(bool)
        self.occ_flag = np.ones(num_points).astype(bool) # mark if a point is set to free space

        self.trunc = trunc

    def fuse(self, depth_im, cam_intr, cam_pose):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # # World coordinates to camera coordinates
        world2cam = np.linalg.inv(cam_pose)
        # world2cam = cam_pose
        cam_pts = np.dot(world2cam[:3, :3], self.world_pts_unalign) + np.tile(world2cam[:3, 3].reshape(3, 1), (1, self.world_pts.shape[1]))

        # Camera coordinates to image pixels
        pix_x = np.round(cam_intr[0, 0] * (cam_pts[0, :] / cam_pts[2, :]) + cam_intr[0, 2]).astype(int)
        pix_y = np.round(cam_intr[1, 1] * (cam_pts[1, :] / cam_pts[2, :]) + cam_intr[1, 2]).astype(int)

        # Skip if outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                                   cam_pts[2, :] > 0))))

        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        depth_diff = cam_pts[2, :] - depth_val
        free_pts = np.logical_and(depth_val > 0, depth_diff < 0)
        if self.trunc is not None: # truncated occupancy
            occ_pts = np.logical_and(depth_val > 0,
                      np.logical_and(depth_diff >= 0, depth_diff < self.trunc))
        else:
            occ_pts = np.logical_and(depth_val > 0, depth_diff >= 0)

        # Remove the points inside view frustum but in the free space
        self._occ_vol[free_pts] = False

        # Only update those occupied points never set to free space even once before
        self.occ_flag[free_pts] = False # update the flag of free-space valid points
        occ_pts = np.logical_and(occ_pts, self.occ_flag)
        self._occ_vol[occ_pts] = True

        print('\n number of occupied voxels: ', np.count_nonzero(self._occ_vol), np.count_nonzero(self.occ_flag))

    # Normalize points to a predefined range
    def normalize_pts(self, padding=0, swap_yz=False):
        vol_center = np.mean(self._vol_bnds, axis=1)
        self.points = (np.transpose(self.world_pts, (1,0)) - vol_center) / self._vol_bound_max # range (-0.5, 0.5)
        # Orientate mesh on z = -0.5
        self.points[:, 2] = (self.points[:, 2] - self.points[:, 2].min()) - 0.5
        
        self.points = (1 + padding) * self.points

        if swap_yz:
            # swap values in x and z directions
            pts = self.points.copy()
            self.points[:, 1] = pts[:, 2]
            self.points[:, 2] = pts[:, 1]

        return self.points, vol_center, self._vol_bound_max, self._occ_vol

    # Save the output
    def save_occ(self, filename, points_dtype=np.float16):
        out_dict = {
                    'points': self.points.astype(points_dtype),
                    'occupancies': np.packbits(self._occ_vol)
                    }
        np.savez(filename, **out_dict)


class OccupancyFusion(object):

    def __init__(self, vol_bnds, voxel_size):

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        # in meters (determines volume discretization and resolution)
        self._voxel_size = voxel_size

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:, 1]-self._vol_bnds[:, 0])/self._voxel_size).copy(
            order='C').astype(int)  # ensure C-order contigous
        self._vol_dim = max(self._vol_dim)
        # create a unit cube based on the maximum coordinates
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(
            order='C').astype(np.float32)  # ensure C-order contigous
        print("Voxel volume size: %d x %d x %d" %
              (self._vol_dim, self._vol_dim, self._vol_dim))

        # Initialize pointers to voxel volume in CPU memory
        self._occ_vol = np.zeros([self._vol_dim, self._vol_dim, self._vol_dim]).astype(bool)
        # self._occ_vol = np.zeros(num_points).astype(bool)
        self.occ_flag = np.ones(self._vol_dim*self._vol_dim*self._vol_dim).astype(bool) # mark if a point is set to free space


    def fuse(self, color_im, depth_im, cam_intr, cam_pose):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(range(self._vol_dim), range(self._vol_dim), range(self._vol_dim), indexing='ij')
        vox_coords = np.concatenate((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)), axis=0).astype(int)

        # Voxel coordinates to world coordinates
        world_pts = self._vol_origin.reshape(-1, 1) + vox_coords.astype(float)*self._voxel_size

        # World coordinates to camera coordinates
        world2cam = np.linalg.inv(cam_pose)
        cam_pts = np.dot(world2cam[:3, :3], world_pts) + np.tile(world2cam[:3, 3].reshape(3, 1), (1, world_pts.shape[1]))

        # Camera coordinates to image pixels
        pix_x = np.round(cam_intr[0, 0] * (cam_pts[0, :] / cam_pts[2, :]) + cam_intr[0, 2]).astype(int)
        pix_y = np.round(cam_intr[1, 1] * (cam_pts[1, :] / cam_pts[2, :]) + cam_intr[1, 2]).astype(int)

        # Skip if outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                                   cam_pts[2, :] > 0))))

        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        depth_diff = depth_val-cam_pts[2, :]
        occ_pts = np.logical_and(depth_val > 0, depth_diff >= 0)

        # Remove the points inside view frustum but in the free space
        valid_pix[occ_pts] = False
        self._occ_vol[vox_coords[0, valid_pix], vox_coords[1, valid_pix], vox_coords[2, valid_pix]] = False

        # Only update those occupied points never set to free space even once before
        self.occ_flag[valid_pix] = False # update the flag of free-space valid points
        occ_pts = np.logical_and(occ_pts, self.occ_flag)
        self._occ_vol[vox_coords[0, occ_pts], vox_coords[1, occ_pts], vox_coords[2, occ_pts]] = True

    def normalize_pts(self, padding=0):
        bound_max = np.max(self._vol_bnds[:, 1]-self._vol_bnds[:, 0])
        vol_center = self._vol_origin + bound_max/2.0
        self.points = (self.world_pts - vol_center) / bound_max # range (-0.5, 0.5)
        self.points = (1 + padding) * self.points

    # Save the output
    def save_occ(self, filename, points_dtype=np.float16):
        out_dict = {
                    'points': self.points.astype(points_dtype),
                    'occupancies': np.packbits(self._occ_vol)
                    }
        np.savez(filename, **out_dict)

