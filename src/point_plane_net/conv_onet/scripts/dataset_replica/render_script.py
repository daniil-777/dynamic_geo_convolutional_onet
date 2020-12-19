# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse
import sys
import os
from math import radians
import bpy
from mathutils import Vector, Color
import colorsys
import numpy as np
import math
import mathutils


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=1,
                    help='number of views to be rendered')
parser.add_argument('--n_aug', type=int, default=1,
                    help='number of augmentations to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--transform', type=str,
                    help='Path to the npy file containing transformation.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--padding', type=float, default=0.05,
                    help='Padding applied to model for data augmentation.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=0.5,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--camera', type=str, default='sphere',
                    help='Wether to randomize camera position [fixed, circle, sphere, general].')
parser.add_argument('--resolution', type=int, default=137,
                    help='Output resultion.')
parser.add_argument('--random_material', action='store_true',
                    help='Wether to randomize material.')
parser.add_argument('--random_size', action='store_true',
                    help='Wether to randomize size of object.')
parser.add_argument('--camera_file', type=str,
                    help='Camera file that should be used')
parser.add_argument('--albedo', action="store_true",
                    help='Whether albedo should be rendered.')
parser.add_argument('--no-cam', action="store_true",
                    help='Whether cameras should not be saved.')
# parser.add_argument('--use-new-camera-convention', action='store_true',
#                     help='Wether to use new camera convention.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


# Create output folder
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

bpy.context.scene.render.use_antialiasing = False

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    # Remap as other types can not represent the full range of depth.
    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [0]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

# Load object
# bpy.ops.import_scene.obj(filepath=args.obj)
bpy.ops.import_mesh.ply(filepath=args.obj)

model = bpy.data.objects.new('Model', None)
for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    object.parent = model
bpy.context.scene.objects.link(model)

# Load transform
if args.transform is not None:
    t0_dict = np.load(args.transform)
    print('\n' * 10, [k for k in t0_dict.keys()], '\n' * 10)
    t0_scale = t0_dict['scale']
    t0_loc = t0_dict['loc']
    # bb0_min, bb0_max = t0_dict['bb0_min'], t0_dict['bb0_max']
    # bb1_min, bb1_max = t0_dict['bb1_min'], t0_dict['bb1_max']
    # print(t0_scale, t0_loc, bb0_max, bb0_min, bb1_min, bb1_max)
else:
    t0_scale = 1. #np.ones(1)
    t0_loc = np.zeros(3)

if args.camera_file is not None:
    cam = np.load(args.camera_file)
    p = np.zeros((4, 1))
    p[-1] = 1
    p_origins = []
    for i in range(args.views):
        p_cami = cam.get('world_mat_inv_%d' % i) @ cam.get('camera_mat_inv_%d' % i) @ p
        p_cami = p_cami[:3, 0]
        p_origins.append(p_cami)
else:
    p_origins = None

# Modifiers
for object in model.children:
    bpy.context.scene.objects.active = object

    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'HEMI'
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = False
lamp.energy = 0.5

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def vec_to_blender(vec):
    v1, v2, v3 = np.split(vec, 3, axis=-1)
    vec = np.concatenate([v1, -v3, v2], axis=-1)
    return vec


def get_random_camera(mode):
    if mode in ('general', 'sphere'):
        if mode == 'general':
            cam_r = np.random.uniform(0.7, 1.5)
        else:
            cam_r = 1.5

        cam_loc = np.zeros(3)
        while np.linalg.norm(cam_loc) <= 1e-2:
            cam_loc = np.random.randn(3)
        cam_loc[2] = abs(cam_loc[2])
        cam_loc = cam_loc * cam_r / np.linalg.norm(cam_loc)
    elif mode == 'circle':
        cam_r = 4
        cam_z = 0.8
        cam_loc2d = np.zeros(2)
        while np.linalg.norm(cam_loc2d) <= 1e-2:
            cam_loc2d = np.random.randn(2)
        cam_loc2d = cam_loc2d * cam_r / np.linalg.norm(cam_loc2d)
        cam_loc = np.array([cam_loc2d[0], cam_loc2d[1], cam_z])
    elif mode == 'inside':
        cam_loc = np.random.uniform(-0.05, 0.05, size=(3,))
        # cam_loc = np.random.uniform(0., 0.0001, size=(3,))
        #cam_loc = np.zeros(3,)
    else:
        raise ValueError('Invalid camera sampling mode "%s"' % mode)

    return cam_loc

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def look_at2(cam, img_idx=0):
    # cam.location = (0, 1.3, 0.8)
    cam.parent = None 

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    # origin = (0, 0.1, 0.01)
    # origin = np.random.rand(3,) - 0.5
    # thetas = np.linspace(0)
    theta = radians(i / 24. * 360)
    phi = radians(90)
    r = 0.1
    x = r * np.sin(phi)*np.cos(theta)
    z = r * np.sin(phi)* np.sin(theta)
    y = r * np.cos(phi)

    origin = np.array([x, y, z])

    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    cam.parent = b_empty  # setup parenting
    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    cam_constraint.target = b_empty



def update_camera(cam, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=.3):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float

    source of function: https://blender.stackexchange.com/questions/100414/how-to-set-camera-location-in-the-scene-while-pointing-towards-an-object-with-a
    """
    looking_direction = cam.location - focus_point
    rot_quat = looking_direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    #camera.location = rot_quat * mathutils.Vector((0.0, 0.0, distance))


# Rendering options
scene = bpy.context.scene
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
# scene.render.alpha_mode = 'TRANSPARENT'
bpy.data.worlds["World"].horizon_color = (1., 1., 1.)

# Set up camera
cam = scene.objects['Camera']
# if args.camera != 'inside':
#     cam.location = (0, 1.3, 0.8)
#     cam_constraint = cam.constraints.new(type='TRACK_TO')
#     cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
#     cam_constraint.up_axis = 'UP_Y'
#     b_empty = parent_obj_to_camera(cam)
#     cam_constraint.target = b_empty

# Lamp constraint
lamp = scene.objects['Lamp']
lamp_constraint = lamp.constraints.new(type='TRACK_TO')
lamp_constraint.track_axis = 'TRACK_NEGATIVE_Z'
lamp_constraint.up_axis = 'UP_Y'
b_empty2 = parent_obj_to_camera(lamp)
lamp_constraint.target = b_empty2

# Some output options
model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(args.output_folder, model_identifier)
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

if args.albedo:
    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'

    links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])
    for output_node in [albedo_file_output]:
        output_node.base_path = ''


for output_node in [depth_file_output]:
    output_node.base_path = ''



default_colors_hsv = {material: material.diffuse_color.hsv
                      for material in bpy.data.materials}

print()

cameras_world = []
cameras_projection = []

blender_T1 = np.array([
    [1., 0., 0, 0.],
    [0., 0., -1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]
])

blender_T2 = np.array([
    [1., 0., 0, 0.],
    [0., -1., 0., 0.],
    [0., 0., -1., 0.],
    [0., 0., 0., 1.]
])


# if args.use_new_camera_convention:
K0 = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
])
# else:
#     K0 = np.array([
#         [0., 1., 0., 0.],
#         [1., 0., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.],
#     ])

assert(args.camera in ('fixed', 'circle', 'sphere', 'general', 'inside'))

t0_scale = 1 / t0_scale
t_scale = [t0_scale, t0_scale, t0_scale]
t_loc = -t0_loc  #trnd_loc[i_aug] + trnd_scale[i_aug] * t0_loc

model.scale = Vector((t_scale[0], t_scale[2], t_scale[1]))
model.location = Vector((t_loc[0], -t_loc[2], t_loc[1]))

# FOR NMR DATASET
camera = bpy.data.cameras.values()[0]
camera.sensor_width = 1
camera.sensor_height = 1
# print('before', camera.lens)
#camera.lens = 1.8660254037844388
camera.lens = 1.

# def look_at(obj_camera, point):
#     loc_camera = obj_camera.matrix_world.to_translation()

#     direction = point - loc_camera
#     # point the cameras '-Z' and use its 'Y' as up
#     rot_quat = direction.to_track_quat('-Z', 'Y')

#     # assume we're using euler rotation
#     obj_camera.rotation_euler = rot_quat.to_euler()

out_dict = {}
for i in range(0, args.views):

    # TODO: should we include to_scale0 and t0_loc here?
    # t_scale = t0_scale # trnd_scale[i_aug] * t0_scale

    # Blender coordinate convention

    # model.location = Vector((0, 0, 0))
    # model.scale = Vector((1, 1, 1))



    # if args.random_material and i != 0:
    #     for material in bpy.data.materials:
    #         h, s, v = default_colors_hsv[material]
    #         h = np.random.uniform((h - 0.2) % 1, (h + 0.2) % 1)
    #         s = np.random.uniform(max(0, s - 0.2), min(1, s + 0.2))
    #         v = np.random.uniform(max(0, v - 0.2), min(1, v + 0.2))
    #         material.diffuse_color.hsv = h, s, v

    print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

    # if p_origins is not None:
    #     p_origini = p_origins[i]
    #     cam.location = Vector((p_origini[0], -p_origini[2], p_origini[1]))
    # else:
    #     if args.camera != 'fixed':
    #         cam_loc = get_random_camera(args.camera)
    #         cam.location = Vector((cam_loc[0], cam_loc[1], cam_loc[2]))
    #     print(cam.location)

    if args.camera == 'inside':

        # bpy.data.objects['Camera'].select = True
        # bpy.ops.object.delete()

        # bpy.ops.object.camera_add()
        # scene.camera = bpy.context.object
        # cam = scene.objects['Camera']

        cam.location = Vector((0, 0, 0))
        # focus_p = (0, -0.1, 0.1)
        # focus_p = (0.00, 0.1, 0.)
        # focus_p = (0.1, 0.0, 0.)

        bpy.context.scene.update()
        # look_at(cam, Vector((focus_p[0], -focus_p[2], focus_p[1])))

        # cam.parent = None 
        # cam_constraint = cam.constraints.new(type='TRACK_TO')
        # cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        # cam_constraint.up_axis = 'UP_Y'
        # origin = (0, 0.1, 0.01)
        # origin = np.random.rand(3,) - 0.5
        # thetas = np.linspace(0)
        theta = radians(i *1.0 / args.views * 360)
        # phi = radians(90)
        # r = 0.1
        # x = r * np.sin(phi)*np.cos(theta)
        # z = r * np.sin(phi)* np.sin(theta)
        # y = r * np.cos(phi)
        cam.rotation_euler = [0,theta,0]
        # origin = np.array([x, y, z])

        # b_empty = bpy.data.objects.new("Empty", None)
        # b_empty.location = origin
        # cam.parent = b_empty  # setup parenting
        # scn = bpy.context.scene
        # scn.objects.link(b_empty)
        # scn.objects.active = b_empty
        # cam_constraint.target = b_empty


        # look_at2(cam, i)
        bpy.context.scene.update()
        #update_camera(cam, mathutils.Vector((focus_p[0], -focus_p[2], focus_p[1])), distance=.1)
    

    # lamp_loc = 2 * get_random_camera('general')
    # lamp.location = Vector((lamp_loc[0], lamp_loc[1], lamp_loc[2]))
    lamp.location = Vector((0, 0, 10))
    bpy.data.lamps['Lamp'].energy = 0.2 + 0.8 #* np.random.rand()
    # bpy.data.objects['Lamp'].rotation_euler[0] = np.random.rand() * 360
    # bpy.data.objects['Sun'].rotation_euler[0] = np.random.rand() * 360

    scene.render.filepath = os.path.join(args.output_folder, 'image', '%04d' % i)
    depth_file_output.file_slots[0].path = os.path.join(args.output_folder, 'depth', '%04d' % i)
    if args.albedo:
        albedo_file_output.file_slots[0].path = os.path.join(args.output_folder, 'albedo', '%04d' % i)

    bpy.ops.render.render(write_still=True)  # render still
    # Save camera properties

    cam_M = np.asarray(cam.matrix_world.inverted())

    # Blender coordinate convention
    cam_M = blender_T2 @ cam_M  @ blender_T1
    cameras_world.append(cam_M)

    cam_P = np.asarray(cam.calc_matrix_camera(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        bpy.context.scene.render.pixel_aspect_x,
        bpy.context.scene.render.pixel_aspect_y,
    ))
    cam_P = K0 @ cam_P
    cam_P = np.vstack([
        cam_P[0], cam_P[1], np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])
    ])

    # R_t = cam_M[:3, :3].T
    # t = cam_M[:3, -1]
    # cam_M_inv = np.eye(4)
    # cam_M_inv[:3, :3] = R_t
    # cam_M_inv[:3, -1] = -R_t @ t
    
    out_dict['camera_mat_%d' % i] = cam_P
    out_dict['camera_mat_inv_%d' % i] = np.linalg.inv(cam_P)
    out_dict['world_mat_%d' % i] = cam_M
    out_dict['world_mat_inv_%d' % i] = np.linalg.inv(cam_M)
    # b_empty.rotation_euler[2] += radians(stepsize)

if not args.no_cam:
    np.savez(os.path.join(args.output_folder, 'cameras.npz'), **out_dict)
