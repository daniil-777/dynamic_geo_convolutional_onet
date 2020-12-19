from im2mesh.encoder import (
    conv, pix2mesh_cond, pointnet, pointnet2,
    psgn_cond, r2n2, voxels, pointnetpp, unet,
)


encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'simple_unet': conv.ConvUnet,
    'unet': unet.UNet,
    'resnet18': conv.Resnet18,
    'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    'r2n2_simple': r2n2.SimpleConv,
    'r2n2_resnet': r2n2.Resnet,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet_resnet_local': pointnet.LocalResnetPointnet,
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'dynamic_pointnet_local_pool': pointnet.DynamicLocalPoolPointnet,
    'dynamic_pointnet_local_pool2': pointnet.DynamicLocalPoolPointnet2,
    'pointnet_local_conv': pointnet2.LocalResnetPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'psgn_cond': psgn_cond.PCGN_Cond,
    'voxel_simple': voxels.VoxelEncoder,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
    'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,
}
