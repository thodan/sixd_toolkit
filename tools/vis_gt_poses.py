# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Visualizes the object models at the ground truth poses.

import os
import sys
import numpy as np
import yaml

sys.path.append(os.path.abspath('..'))
from pysixdb import inout, misc, renderer
from params import par_hinterstoisser as par

# Path mask for output images
vis_mpath = '../output/vis_gt_poses/{:02d}_{:04d}.png'

misc.ensure_dir(os.path.dirname(vis_mpath.format(0, 0)))

scene_ids = range(1, 16)
scene_im_ids = range(0, 1000, 100)

for scene_id in scene_ids:
    # Load object model
    obj_id = scene_id  # The object id is the same as scene id here
    model = inout.load_ply(par.model_mpath.format(obj_id))

    # Load scene info and gt poses
    with open(par.scene_info_mpath.format(scene_id), 'r') as f:
        scene_info = yaml.load(f, Loader=yaml.CLoader)
    with open(par.scene_gt_mpath.format(scene_id), 'r') as f:
        scene_gt = yaml.load(f, Loader=yaml.CLoader)

    for im_id in scene_im_ids:
        print('scene,view: ' + str(scene_id) + ',' + str(im_id))

        # Load the images
        rgb = inout.read_im(par.test_rgb_mpath.format(scene_id, im_id))
        #depth = inout.read_depth(depth_mpath.format(scene_id, im_id)) # [100um]

        vis_rgb = np.zeros(rgb.shape, np.float32)
        for gt in scene_gt[im_id]:
            ren_rgb = renderer.render(model, par.cam['im_size'], par.cam['K'],
                                      np.array(gt['cam_R_m2c']).reshape((3, 3)),
                                      np.array(gt['cam_t_m2c']).reshape((3, 1)),
                                      mode='rgb')
            ren_rgb = misc.draw_rect(ren_rgb.astype(np.uint8), gt['obj_bb'])
            vis_rgb += ren_rgb.astype(np.float32)

        vis_rgb = 0.6 * vis_rgb.astype(np.float32) + 0.4 * rgb.astype(np.float32)
        vis_rgb[vis_rgb > 255] = 255

        inout.write_im(vis_mpath.format(scene_id, im_id), vis_rgb.astype(np.uint8))
