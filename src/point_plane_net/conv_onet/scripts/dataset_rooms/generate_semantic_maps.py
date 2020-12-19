import numpy as np
from os import listdir
from os.path import join, isdir
from PIL import Image
from tqdm import tqdm


ds_path = '/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/RoomDataset5'
resolution = 256
object_classes = ['04256520', '03636649', '03001627', '04379243', '02933112']
object_classes.sort()
item_dict_name = 'item_dict.npz'
map_scale = (-0.5, 0.5)
ground_plane_channel = len(object_classes)
wall_channel = len(object_classes) + 1

def get_visualization_for_semantic_map(semantic_map):
    # semantic map of shape resolution x resolution x len(object_classes)
    semantic_image = (255 * np.ones((resolution, resolution, 3))).astype(np.uint8)
    
    # add ground plane
    semantic_image[semantic_map[:, :, ground_plane_channel] == 1] = [100, 100, 100]

    # add walls
    semantic_image[semantic_map[:, :, wall_channel] == 1] = [0, 0, 0]
    
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]
    for cl_idx in range(len(object_classes)):
        mask = semantic_map[:, :, cl_idx] == 1
        color = colors[cl_idx]
        semantic_image[mask] = color
    semantic_image = Image.fromarray(semantic_image).convert("RGB")
    return semantic_image


def add_ground_plane(semantic_map, ground_plane_scales):
    offset = ((1. - ground_plane_scales) / 2.)
    offset = (offset * resolution).astype(int)
    if (offset[0] == 0) and (offset[1] == 0):
        semantic_map[:, :, ground_plane_channel] = 1.
    elif offset[0] == 0:
        semantic_map[:, offset[1]:-offset[1], ground_plane_channel] = 1.
    elif offset[1] == 0:
        semantic_map[offset[0]:-offset[0], :, ground_plane_channel] = 1.
    else:
        semantic_map[offset[0]:-offset[0], offset[1]:-offset[1], ground_plane_channel] = 1.
    return semantic_map


def add_walls(semantic_map, walls, ground_plane_scales, wall_thickness=0.01):
    wall_thickness = int(wall_thickness * resolution)
    if walls[0] == 1:
        offset_x = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[0]) * resolution / 2)
        offset_y1 = 0 if ground_plane_scales[1] == 1 else int((1 - ground_plane_scales[1]) * resolution / 2)
        if offset_x == 0:
            semantic_map[:, offset_y1:offset_y1+wall_thickness, wall_channel] = 1
        else:
            semantic_map[offset_x:-offset_x, offset_y1:offset_y1+wall_thickness, wall_channel] = 1
    if walls[1] == 1:
        offset_y = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[1]) * resolution / 2)
        offset_x1 = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[0]) * resolution / 2)
        if offset_y == 0:
            semantic_map[offset_x1:offset_x1+wall_thickness, :, wall_channel] = 1
        else:
            semantic_map[offset_x1:offset_x1+wall_thickness, offset_y:-offset_y, wall_channel] = 1
    if walls[2] == 1:
        offset_x = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[0]) * resolution / 2)
        # offset_y1 = 0 if ground_plane_scales[1] == 1 else int((1 - ground_plane_scales[1]) * resolution / 2)
        if offset_x == 0:
            semantic_map[:, -wall_thickness:, wall_channel] = 1
        else:
            semantic_map[offset_x:-offset_x, -wall_thickness:, wall_channel] = 1
    if walls[3] == 1:
        offset_y = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[1]) * resolution / 2)
        # offset_x1 = 0 if ground_plane_scales[0] == 1 else int((1 - ground_plane_scales[0]) * resolution / 2)
        if offset_y == 0:
            semantic_map[-wall_thickness:, :, wall_channel] = 1
        else:
            semantic_map[-wall_thickness, offset_y:-offset_y, wall_channel] = 1
    return semantic_map


classes = [c for c in listdir(ds_path) if isdir(join(ds_path, c))]
classes.sort()
for cl in classes:
    print("Processing class %s ..." % cl)
    cl_dir = join(ds_path, cl)
    models = [m for m in listdir(cl_dir) if isdir(join(cl_dir, m))]
    models.sort()
    for model in tqdm(models):
        model_path = join(cl_dir, model)
        item_dict = np.load(join(model_path, item_dict_name), allow_pickle=True)

        # make output map
        semantic_map = np.zeros((resolution, resolution, len(object_classes) + 1 + 1)) # for ground plane / walls
        item_bboxes = item_dict['bboxes']
        item_classes = item_dict['classes']
        # import ipdb; ipdb.set_trace()
        # Add objects
        for item_idx, item_class in enumerate(item_classes):
            # get class of item
            cl_idx = object_classes.index(item_class)
            # get pixel locations of item 
            item_bbox = ((resolution - 1) * (item_bboxes[item_idx] - map_scale[0]) / (map_scale[1] - map_scale[0])).astype(int)
            semantic_map[item_bbox[0,0]:item_bbox[1, 0], item_bbox[0, 1]:item_bbox[1, 1], cl_idx] = 1
        # Add Ground plane
        ground_plane_scales = item_dict['xz_groundplane_range']
        semantic_map = add_ground_plane(semantic_map, ground_plane_scales)

        # add walls
        walls = item_dict['walls']
        semantic_map = add_walls(semantic_map, walls, ground_plane_scales)

        out_file = join(model_path, 'semantic_map.npz')
        np.savez(out_file, semantic_map=semantic_map)
        out_file = join(model_path, 'semantic_map.jpg')
        get_visualization_for_semantic_map(semantic_map).save(out_file)