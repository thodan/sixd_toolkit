# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Visualizes the object models at the ground truth poses.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from pysixdb import inout, misc, renderer

# Dataset parameters
# from params import par_hinterstoisser as par
# from params import par_tejani as par
from params import par_doumanoglou as par

# Select IDs of scenes, images and GT poses to be processed.
# Empty list [] means that all IDs will be used.
scene_ids = []
im_ids = []
gt_ids = []

# Indicates whether to render RGB image
vis_rgb = True

# Indicates whether to resolve visibility in the rendered RGB image (using
# depth renderings). If True, only the part of object surface, which is not
# occluded by any other modeled object, is visible. If False, RGB renderings
# of individual objects are blended together.
vis_rgb_resolve_visib = True

# Indicates whether to render depth image
vis_depth = True

# Path masks for output images
vis_rgb_mpath = '../output/vis_gt_poses/{:02d}_{:04d}.jpg'
vis_depth_mpath = '../output/vis_gt_poses/{:02d}_{:04d}_depth_diff.jpg'
misc.ensure_dir(os.path.dirname(vis_rgb_mpath))

scene_ids_curr = range(1, par.scene_count + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids)
for scene_id in scene_ids_curr:
    # Load scene info and gt poses
    scene_info = inout.load_scene_info(par.scene_info_mpath.format(scene_id))
    scene_gt = inout.load_scene_gt(par.scene_gt_mpath.format(scene_id))

    # Load models of objects that appear in the current scene
    obj_ids = set([gt['obj_id'] for gts in scene_gt.values() for gt in gts])
    models = {}
    for obj_id in obj_ids:
        models[obj_id] = inout.load_ply(par.model_mpath.format(obj_id))

    # Visualize GT poses in the selected images
    im_ids_curr = sorted(scene_info.keys())
    if im_ids:
        im_ids_curr = set(im_ids_curr).intersection(im_ids)
    for im_id in im_ids_curr:
        print('scene: {}, im: {}'.format(scene_id, im_id))

        # Load the images
        rgb = inout.read_im(par.test_rgb_mpath.format(scene_id, im_id))
        depth = inout.read_depth(par.test_depth_mpath.format(scene_id, im_id))
        depth = depth.astype(np.float) * 0.1 # [mm]

        # Render the objects at the ground truth poses
        im_size = (depth.shape[1], depth.shape[0])
        ren_rgb = np.zeros(rgb.shape, np.float)
        ren_depth = np.zeros(depth.shape, np.float)

        gt_ids_curr = range(len(scene_gt[im_id]))
        if gt_ids:
            gt_ids_curr = set(gt_ids_curr).intersection(gt_ids)
        for gt_id in gt_ids_curr:
            gt = scene_gt[im_id][gt_id]

            model = models[gt['obj_id']]
            K = par.cam['K']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']

            # Rendering
            if vis_rgb:
                m_rgb, m_depth = renderer.render(model, im_size, K, R, t,
                                                 mode='rgb+depth')
            if vis_depth or (vis_rgb and vis_rgb_resolve_visib):
                m_depth = renderer.render(model, im_size, K, R, t, mode='depth')

                # Get mask of the surface parts that are closer than the
                # surfaces rendered before
                visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
                mask = np.logical_and(m_depth != 0, visible_mask)

                ren_depth[mask] = m_depth[mask].astype(ren_depth.dtype)

            # Combine the RGB renderings
            if vis_rgb:
                if vis_rgb_resolve_visib:
                    ren_rgb[mask] = m_rgb[mask].astype(ren_rgb.dtype)
                else:
                    ren_rgb += m_rgb.astype(ren_rgb.dtype)

        # Save RGB visualization
        if vis_rgb:
            vis_im_rgb = 0.4 * rgb.astype(np.float) + 0.6 * ren_rgb
            vis_im_rgb[vis_im_rgb > 255] = 255
            inout.write_im(vis_rgb_mpath.format(scene_id, im_id),
                           vis_im_rgb.astype(np.uint8))

        # Save image of depth differences
        if vis_depth:
            # Calculate the depth difference at pixels where both depth maps
            # are valid
            valid_mask = (depth > 0) * (ren_depth > 0)
            depth_diff = valid_mask * (depth - ren_depth.astype(np.float))

            plt.matshow(depth_diff)
            plt.axis('off')
            plt.title('measured - GT depth [mm]')
            plt.colorbar()
            plt.savefig(vis_depth_mpath.format(scene_id, im_id), pad=0,
                        bbox_inches='tight')
            plt.close()
