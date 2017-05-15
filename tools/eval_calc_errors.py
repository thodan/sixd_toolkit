# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates error of 6D object pose estimates.

import os
import sys
import glob
#import time

sys.path.append(os.path.abspath('..'))
from pysixd import inout, pose_error, misc
from params.dataset_params import get_dataset_params

# Paths
#-------------------------------------------------------------------------------
# Results for which the errors will be calculated
result_base = '/home/tom/th_data/cmp/projects/sixd/sixd_results/'
result_paths = [
    result_base + 'hodan-iros15-forwacv17_tless_primesense'
]

# Mask of path to the output file with calculated errors
errs_mpath = '{result_path}_eval/{scene_id:02d}_{err_desc}.txt'

# Parameters
#-------------------------------------------------------------------------------
# Top N pose estimates (according to their score) to be evaluated for each
# object in each image
top_n_ests = 1 # None to consider all estimates

# Pose error function
error_type = 'vsd' # 'vsd', 'adi', 'add', 'cou', 're', 'te'

# VSD parameters
delta = 15
tau = 20

err_desc = 'error=' + error_type
if error_type == 'vsd':
    err_desc += '-delta=' + str(delta) + '-tau=' + str(tau)

# Error calculation
#-------------------------------------------------------------------------------
for result_path in result_paths:
    info = os.path.basename(result_path).split('_')
    method = info[0]
    dataset = info[1]
    test_type = info[2] if len(info) > 2 else ''

    # Select a type of the 3D object model
    if dataset == 'tless':
        model_type = 'cad'
        cam_type = test_type
    else:
        model_type = ''
        cam_type = ''

    # Load dataset parameters
    dp = get_dataset_params(dataset, model_type=model_type, test_type=test_type,
                            cam_type=cam_type)

    # Load object models
    if error_type in ['vsd', 'add', 'adi', 'cou']:
        print('Loading object models...')
        models = {}
        for obj_id in range(1, dp['obj_count'] + 1):
            models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

    # Directories with results for individual scenes
    scene_dirs = sorted([d for d in glob.glob(os.path.join(result_path, '*'))
                         if os.path.isdir(d)])

    for scene_dir in scene_dirs:
        scene_id = int(os.path.basename(scene_dir))

        # Load info and GT poses for the current scene
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

        res_paths = sorted(glob.glob(os.path.join(scene_dir, '*.txt')))
        errs = []
        im_id = -1
        depth_im = None
        for res_path in res_paths:
            #t = time.time()

            # Parse image ID and object ID from the file name
            filename = os.path.basename(res_path).split('.')[0]
            im_id_prev = im_id
            im_id, obj_id = map(int, filename.split('_'))

            print('Calculating {} error - scene: {}, im: {}, obj: {}'.format(
                error_type, scene_id, im_id, obj_id))

            # Load depth image if VSD is selected
            if error_type == 'vsd':
                if im_id != im_id_prev:
                    depth_path = dp['test_depth_mpath'].format(scene_id, im_id)
                    depth_im = inout.read_depth(depth_path)
                    depth_im *= dp['cam']['depth_scale'] # to [mm]

            # Load camera matrix
            if error_type in ['vsd', 'cou']:
                K = scene_info[im_id]['cam_K']

            # Load pose estimates
            ests = inout.load_poses(res_path)

            # Sort the estimates by score (in descending order)
            ests_sorted = sorted(enumerate(ests), key=lambda x: x[1]['score'],
                                 reverse=True)

            # Consider only the top N estimated poses
            ests_sorted = ests_sorted[slice(0, top_n_ests)]

            model = models[obj_id]
            for est_id, est in ests_sorted:
                est_errs = []
                R_est = est['cam_R_m2c']
                t_est = est['cam_t_m2c']

                for gt_id, gt in enumerate(scene_gt[im_id]):
                    if gt['obj_id'] != obj_id:
                        continue

                    err = -1.0
                    R_gt = gt['cam_R_m2c']
                    t_gt = gt['cam_t_m2c']

                    if error_type == 'vsd':
                        err = pose_error.vsd(R_est, t_est, R_gt, t_gt, model,
                                             depth_im, delta, tau, K)
                    elif error_type == 'add':
                        err = pose_error.add(R_est, t_est, R_gt, t_gt, model)
                    elif error_type == 'adi':
                        err = pose_error.adi(R_est, t_est, R_gt, t_gt, model)
                    elif error_type == 'cou':
                        err = pose_error.cou(R_est, t_est, R_gt, t_gt, model,
                                             dp['test_im_size'], K)
                    elif error_type == 're':
                        err = pose_error.re(R_est, R_gt)
                    elif error_type == 'te':
                        err = pose_error.te(t_est, t_gt)

                    errs.append([im_id, obj_id, est_id, gt_id, err])
            #print('Evaluation time: {}s'.format(time.time() - t))

        print('Saving calculated errors...')
        errs_path = errs_mpath.format(result_path=result_path,
                                      scene_id=scene_id,
                                      err_desc=err_desc)
        misc.ensure_dir(os.path.dirname(errs_path))
        inout.save_errors_sixd2017(errs_path, errs)
