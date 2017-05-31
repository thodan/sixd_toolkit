# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Estimates visibility of object models at the ground truth poses.

import os
import sys
import numpy as np

sys.path.append(os.path.abspath('..'))
from pysixd import inout, misc, renderer, visibility
from params.dataset_params import get_dataset_params

# dataset = 'hinterstoisser'
dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'

delta = 15 # Tolerance used in the visibility test [mm]
do_vis = True # Whether to save visualizations of visibility masks

# Select data type
if dataset == 'tless':
    test_type = 'primesense'
    cam_type = 'primesense'
    model_type = 'cad'
else:
    test_type = ''
    model_type = ''
    cam_type = ''

# Load dataset parameters
dp = get_dataset_params(dataset, model_type=model_type, test_type=test_type,
                        cam_type=cam_type)

# Mask of path of the output visualizations
vis_mpath = '../output/visib_{}_delta={}/{:02d}_{:' +\
            str(dp['im_id_pad']).zfill(2) + 'd}_{:02d}.jpg'
misc.ensure_dir(os.path.dirname(vis_mpath))

print('Loading object models...')
models = {}
for obj_id in range(1, dp['obj_count'] + 1):
    models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

for scene_id in range(1, dp['scene_count'] + 1):

    # Load scene info and gts
    info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
    gts = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

    # Estimate visibility of all ground truth poses in each test image
    im_ids = sorted(gts.keys())
    visib_fracts = {}
    for im_id in im_ids:
        print('dataset: {}, scene: {}, im: {}'.format(dataset, scene_id, im_id))

        K = info[im_id]['cam_K']
        depth_path = dp['test_depth_mpath'].format(scene_id, im_id)
        depth_im = inout.load_depth(depth_path)
        depth_im *= dp['cam']['depth_scale'] # to [mm]
        im_size = (depth_im.shape[1], depth_im.shape[0])

        # Estimate visibility
        visib_fracts[im_id] = []
        for gt_id, gt in enumerate(gts[im_id]):
            depth_gt = renderer.render(models[gt['obj_id']], im_size, K,
                                       gt['cam_R_m2c'], gt['cam_t_m2c'],
                                       mode='depth')
            dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
            dist_im = misc.depth_im_to_dist_im(depth_im, K)

            # Estimation of visibility mask
            visib_gt = visibility.estimate_visib_mask_gt(dist_im, dist_gt, delta)

            # Visible surface fraction
            obj_mask_gt = dist_gt > 0
            visib_fract = visib_gt.sum() / float(obj_mask_gt.sum())
            visib_fracts[im_id].append(float(visib_fract))

            if do_vis:
                depth_im_vis = misc.norm_depth(depth_im, 0.2, 1.0)
                depth_im_vis = np.dstack([depth_im_vis] * 3)

                visib_gt_vis = visib_gt.astype(np.float)
                zero_ch = np.zeros(visib_gt_vis.shape)
                visib_gt_vis = np.dstack([zero_ch, visib_gt_vis, zero_ch])

                vis = 0.5 * depth_im_vis + 0.5 * visib_gt_vis
                vis[vis > 1] = 1
                vis_path = vis_mpath.format(dataset, delta, scene_id,
                                            im_id, gt_id)
                inout.save_im(vis_path, vis)

    res_path = dp['scene_gt_visib_mpath'].format(scene_id, delta)
    misc.ensure_dir(os.path.dirname(res_path))
    #inout.save_yaml(res_path, visib_fracts)
