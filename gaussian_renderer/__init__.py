#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from ctypes import c_char
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import numpy as np
import os

def concrete_dropout_mask(uncertainty, temperature=0.1):
    eps = 1e-7  # numerical stability
    u = torch.rand_like(uncertainty)
    drop_logit = (
        torch.log(uncertainty + eps)
        - torch.log(1 - uncertainty + eps)
        + torch.log(u + eps)
        - torch.log(1 - u + eps)
    )
    z = 1 - torch.sigmoid(drop_logit / temperature)

    return z  # ∈ (0,1)





def process_ugod_opacity(opacity, factor):
    return (1 - factor) * opacity

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
   
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing = pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity


    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    cc = viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    #print(len(cc))
    dir_pp = (pc.get_xyz - cc)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    r = pc.get_rotation
    s = pc.get_scaling


    importance = pc._vdgs(dir_pp_normalized, means3D, r, s)


    factor=importance
    factor.clamp(0,1)


    shs, opacity = process_vdgs(pc, shs, opacity, factor)





    drop_mask = concrete_dropout_mask(factor)


    drop_mask = drop_mask.clamp(0.2, 0.8)

    #this step we save the uncertainty and drop mask to analyse the uncertainty and drop probability
    import numpy as np
    np.save("factor.npy", factor.detach().cpu().numpy())
    np.save("drop_mask.npy", drop_mask.detach().cpu().numpy())


    #
    opacity = opacity * drop_mask
    #print(opacity)




              
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    '''
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    '''


    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
