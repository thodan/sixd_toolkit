# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates error of pose estimates.

import os
import sys
import glob
import yaml

sys.path.append(os.path.abspath('..'))
from pysixd import inout, pose_error
from params.dataset_params import get_dataset_params

# Paths
#-------------------------------------------------------------------------------
# Sets of results to be evaluated
result_paths = [
    'results/samplemethod_tless_primesense'
]

# Mask of path to the output file with calculated errors
errs_mpath = '{result_path}_eval/{scene_id:02d}_{eval_desc}.yml'

# Parameters
#-------------------------------------------------------------------------------
# Top N pose estimates (according to their score) to be evaluated for each
# object in each image
top_n_ests = 1

# Pose error function
pose_error_fun = 'vsd' # 'vsd', 'adi', 'add', 'cou', 'te', 're'
delta = 15 # for VSD
tau = 20 # for VSD

eval_desc = 'error=' + pose_error_fun
if pose_error_fun == 'vsd':
    eval_desc += '_delta=' + str(delta) + '_tau=' + str(tau)
    #eval_desc += '_tau=' + str(tau)

# Error calculation
#-------------------------------------------------------------------------------
for result_path in result_paths:
    info = os.path.basename(result_path).split('_')
    method = info[0]
    dataset = info[1]
    test_type = info[2] if len(info) > 2 else ''
    path = result_path

    # Select a type of the 3D object model for evaluation
    if dataset == 'tless':
        model_type = 'cad'
    else:
        model_type = ''

    # Load dataset parameters
    dp = get_dataset_params(dataset, model_type=model_type, test_type=test_type)

    # Load object models
    models = {}
    for obj_id in range(1, dp['obj_count'] + 1):
        models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

    # Directories with results for individual scenes
    scene_dirs = glob.glob(os.path.join(result_path, '*'))

    for scene_dir in scene_dirs:
        scene_id = int(os.path.basename(scene_dir))

        # Load info and GT poses for the current scene
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

        res_files = glob.glob(os.path.join(scene_dir, '*.txt'))
        errs = {}
        for res_file in res_files:
            # Parse image ID and object ID from the file name
            im_id, obj_id = os.path.basename(res_file).split('.')[0].split('_')

            # Load depth image if VSD is used for evaluation
            if pose_error_fun == 'vsd':
                depth_im = inout.read_depth(dp['test_depth_mpath'].format(scene_id, im_id))
                depth_im *= dp['cam']['depth_scale'] # to [mm]

            # Load camera matrix for image im_id
            K = scene_info[im_id]['cam_K']

            # Get GT poses of object obj_id in image im_id
            gts = [gt for gt in scene_gt[im_id] if gt['obj_id'] == obj_id]

            # Load pose estimates of object obj_id in image im_id
            ests = inout.load_poses(res_file)

            # Sort the estimates by score
            ests = sorted(ests, key=lambda x: x['score'])

            # Consider only the top N estimated poses
            ests = ests[:top_n_ests]

            errs.setdefault(im_id, {}).setdefault(obj_id, [])
            for est in ests:
                est_errs = []
                for gt in gts:
                    err = -1.0
                    if pose_error_fun == 'vsd':
                        err = pose_error.vsd(est, gt, models[obj_id],
                                             depth_im, delta, tau, K)
                    elif pose_error_fun == 'add':
                        err = pose_error.add(est, gt, models[obj_id])
                    elif pose_error_fun == 'adi':
                        err = pose_error.adi(est, gt, models[obj_id])
                    elif pose_error_fun == 'cou':
                        err = pose_error.cou(est, gt, models[obj_id],
                                             dp['test_im_size'], K)
                    elif pose_error_fun == 'te':
                        err = pose_error.te(est['cam_t_m2c'], gt['cam_t_m2c'])
                    elif pose_error_fun == 're':
                        err = pose_error.re(est['cam_R_m2c'], gt['cam_R_m2c'])

                    est_errs.append(err)
                errs[im_id][obj_id].append(est_errs)

        print('Saving calculated errors...')
        errs_path = errs_mpath.format(result_path=result_path, scene_id=scene_id,
                                      eval_desc=eval_desc)
        with open(errs_path, 'w') as f:
            yaml.dump(errs, f, width=10000, Dumper=yaml.CDumper)
