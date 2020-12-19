import torch
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import os
from im2mesh.dvr_common import (
    arange_pixels, transform_to_camera_space)
import matplotlib.image as mpimg
from im2mesh.dvr_common import make_3d_grid


class Renderer(object):
    '''  Render class for Occupancy Networks.

    It provides functions to render the representation.

    Args:
        model (nn.Module): trained Occupancy Network model
        threshold (float): threshold value
        device (device): pytorch device
        colors (string): which type of color to use (default: rgb)
        resolution (tuple): output resolution
        n_views (int): number of views to generate
        extension (string): output image extension
        background (string): which background color to use
    '''

    def __init__(self, model, threshold=0.5, device=None, colors='rgb',
                 resolution=(128, 128), n_views=3, extension='png',
                 background='white'):
        self.model = model.to(device)
        self.threshold = threshold
        self.device = device
        self.colors = colors
        self.n_views = n_views
        self.extension = extension
        self.resolution = resolution

        if background == 'white':
            self.background = 1.
        elif background == 'black':
            self.background = 0.
        else:
            self.background = 0.

    def render_and_export(self, data, img_out_path, modelname='model0',
                          return_stats=True):
        ''' Renders and exports for provided camera information in data.

        Args:
            data (tensor): data tensor
            img_out_path (string): output path
            modelname (string): name of the model
            return_stats (bool): whether stats should be returned
        '''

        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        c = self.model.encode_inputs(inputs)
        z = self.model.get_z_from_prior((1,)).to(device)
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path)
        out_imgs = []
        for i in tqdm(range(10, self.n_views + 10)):
            datai = data.get('img.img%d' % i, None)
            if datai is None:
                print('No image %d found.' % i)
                break
            img = datai[None]
            batch_size, _, h, w = img.shape
            assert(batch_size == 1)
            world_mat = datai.get('world_mat').to(device)
            camera_mat = datai.get('camera_mat').to(device)
            scale_mat = datai.get('scale_mat').to(device)
            t0 = time.time()
            img = self.render_img(camera_mat, world_mat, inputs, scale_mat, z,
                                  c, stats_dict, resolution=self.resolution)
            stats_dict['time_render'] = time.time() - t0
            img.save(os.path.join(
                img_out_path, '%s_%03d.%s' % (modelname, i, self.extension)))
            out_imgs.append(img)
        return out_imgs, stats_dict

    def render_img(self, camera_mat, world_mat, inputs, scale_mat=None, z=None,
                   c=None, stats_dict={}, resolution=(128, 128)):
        ''' Renders an image for provided camera information.

        Args:
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): statistics dictionary
            resolution (tuple): output image resolution
        '''
        device = self.device
        h, w = resolution

        t0 = time.time()

        p_loc, pixels = arange_pixels(resolution=(h, w))
        pixels = pixels.to(device)
        stats_dict['time_prepare_points'] = time.time() - t0

        if self.colors in ('rgb', 'depth'):
            # Get predicted world points
            with torch.no_grad():
                t0 = time.time()
                p_world_hat, mask_pred, mask_zero_occupied = \
                    self.model.pixels_to_world(pixels, camera_mat, world_mat,
                                               scale_mat, z, c,
                                               sampling_accuracy=[1024, 1025])
                stats_dict['time_eval_depth'] = time.time() - t0

            t0 = time.time()
            p_loc = p_loc[mask_pred]
            with torch.no_grad():
                if self.colors == 'rgb':
                    img_out = (255 * np.ones((h, w, 3))).astype(np.uint8)

                    t0 = time.time()
                    if mask_pred.sum() > 0:
                        rgb_hat = self.model.decode_color(p_world_hat, c=c,
                                                          z=z)
                        rgb_hat = rgb_hat[mask_pred].cpu().numpy()
                        rgb_hat = (rgb_hat * 255).astype(np.uint8)
                        img_out[p_loc[:, 1], p_loc[:, 0]] = rgb_hat
                    img_out = Image.fromarray(img_out).convert('RGB')
                elif self.colors == 'depth':
                    img_out = (255 * np.ones((h, w))).astype(np.uint8)
                    if mask_pred.sum() > 0:
                        p_world_hat = p_world_hat[mask_pred].unsqueeze(0)
                        d_values = transform_to_camera_space(
                            p_world_hat, camera_mat, world_mat,
                            scale_mat).squeeze(0)[:, -1].cpu().numpy()
                        m = d_values[d_values != np.inf].min()
                        M = d_values[d_values != np.inf].max()
                        d_values = 0.5 + 0.45 * (d_values - m) / (M - m)
                        d_image_values = d_values * 255
                        img_out[p_loc[:, 1], p_loc[:, 0]] = \
                            d_image_values.astype(np.uint8)
                    img_out = Image.fromarray(img_out).convert("L")

        stats_dict['time_eval_color'] = time.time() - t0
        return img_out

    def run_mayavi_visualization(self, res=128):
        ''' Runs mayavi visualization.

        Args:
            res (int): voxel resolution
        '''
        device = self.device
        shape = (res, res, res)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(1, *p.size())
        occ = []
        with torch.no_grad():
            for pi in torch.split(p, 100000, dim=1):
                occ.append(self.model.decode(pi).probs[0])
        occ = torch.cat(occ)
        occ = occ.view(res, res, res).cpu().numpy()
        np.savez('./occ.npz', occ=occ)
        return 0

    def pilImageToColormap(self, img, mask=None):
        ''' Converts PIL Image to colormap.

        Args:
            img (PIL Image): PIL image
            mask (array): If not None, it specifies which pixels should be
                masked
        '''
        cm_hot = mpimg.cm.get_cmap('viridis')
        im = np.array(img)
        im = cm_hot(im)
        if mask is not None:
            im[mask == 0] = 1.
        im = np.uint8(im * 255)
        im = Image.fromarray(im).convert("RGB")
        return im

    def export(self, img_list, img_out_path, modelname='model0'):
        ''' Exports the image list.

        Args:
            img_list (list): list of images
            img_out_path (string): output path
            modelname (string): model name
        '''
        model_path = os.path.join(img_out_path, modelname)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for i in range(self.n_views):
            out_file = os.path.join(model_path, '%06d.png' % i)
            img_list[i].save(out_file)
        return 0

    def estimate_colors(self, vertices, z=None, c=None):
        ''' Estimates the colors for provided vertices.

        Args:
            vertices (Numpy array): mesh vertices
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        colors = []
        for vi in vertices_split:
            vi = vi.to(device)
            with torch.no_grad():
                ci = self.model.decode_color(vi, z, c).squeeze(0).cpu()
            colors.append(ci)

        colors = np.concatenate(colors, axis=0)
        colors = np.clip(colors, 0, 1)
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([
            colors,
            np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        return colors
