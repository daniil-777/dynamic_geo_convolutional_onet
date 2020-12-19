from im2mesh.encoder import (
    conv, pix2mesh_cond, pointnet,
    psgn_cond, r2n2, voxels, point_plane_net, fc_point_net, additional_encoders
)

## pointplanenet_resnet authors: Daniil Emtsev and Dusan Svilarkovic
encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'resnet18': conv.Resnet18,
    'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    'r2n2_simple': r2n2.SimpleConv,
    'r2n2_resnet': r2n2.Resnet,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'psgn_cond': psgn_cond.PCGN_Cond,
    'voxel_simple': voxels.VoxelEncoder,
    'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,
    'fc_plane_net' : fc_point_net.ResnetPointnet,
    'fc_height_plane_net': fc_point_net.ResnetPointHeightnet,
    'sphere_net': fc_point_net.ResnetSpherePointnet,
    'multi_sphere_net': fc_point_net.ResnetMultiSpherePointnet,
    'sphere_height_net': fc_point_net.ResnetSpherePointHeightnet,
    'voxel_sphere_2d_unet': fc_point_net.ResnetVoxelSpherePointHeightnet,
    'multi_voxels_sphere': fc_point_net.ResnetMultiVoxelSpherePointHeightnet,
    'const_multi_spheres': additional_encoders.ResnetConstMultiSpherePointnet,
    'rotation_plane_net': additional_encoders.ResnetRotationPointnet,
    'rotation_plane_height_net': additional_encoders.ResnetRotationPointHeightnet,
    'sphere_height_plane_net': additional_encoders.ResnetSpherePlaneHeightnet,
    "triplet_fc_plane_net": additional_encoders.ResnetPointnetTriplet
}
