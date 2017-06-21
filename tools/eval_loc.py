# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates performance scores for the 6D localization task.

# For evaluation of the SIXD Challenge 2017 task (6D localization of a single
# instance of a single object), use this setting (TENTATIVE):
# n_top = 1
# error_type = 'vsd'
# vsd_delta = 15
# vsd_tau = 20

import os
import sys
from collections import defaultdict
import numpy as np

sys.path.append(os.path.abspath('..'))
from pysixd import inout, pose_matching
from params.dataset_params import get_dataset_params

# Paths to pose errors (calculated using eval_calc_errors.py)
#-------------------------------------------------------------------------------
error_bpath = '/home/tom/th_data/cmp/projects/sixd/sixd_results/'
# error_bpath = '/datagrid/6DB/sixd_results/'
error_paths = [
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=add_ntop=1',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=adi_ntop=1',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=cou_ntop=1',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=re_ntop=1',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=te_ntop=1',
    error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=vsd_ntop=1_delta=15_tau=20_cost=step',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=vsd_ntop=1_delta=15_tau=30_cost=step',
    # error_bpath + 'hodan-iros15-forwacv17_tless_primesense_eval/error=vsd_ntop=1_delta=15_tau=50_cost=step',

    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=add_ntop=1',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=adi_ntop=1',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=cou_ntop=1',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=re_ntop=1',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=te_ntop=1',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=vsd_ntop=1_delta=15_tau=20_cost=step',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=vsd_ntop=1_delta=15_tau=30_cost=step',
    # error_bpath + 'hodan-iros15-nopso_hinterstoisser_eval/error=vsd_ntop=1_delta=15_tau=50_cost=step',
]

# Other paths
#-------------------------------------------------------------------------------
# Mask of path to the input file with calculated errors
errors_mpath = '{error_path}/errors_{scene_id:02d}.yml'

# Mask of path to the output file with established matches and calculated scores
matches_mpath = '{error_path}/matches_{eval_sign}.yml'
scores_mpath = '{error_path}/scores_{eval_sign}.yml'

# Parameters
#-------------------------------------------------------------------------------
visib_gt_min = 0.1 # Minimum visible surface fraction of valid GT pose
visib_delta = 15 # [mm]

# Threshold of correctness
error_thresh = {
    'vsd': 0.5,
    'cou': 0.5,
    'te': 5.0, # [cm]
    're': 5.0 # [deg]
}

# Factor k; threshold of correctness = k * d, where d is the object diameter
error_thresh_fact = {
    'add': 0.1,
    'adi': 0.1
}

# Matching estimated poses to the GT poses
#-------------------------------------------------------------------------------
for error_path in error_paths:
    print('Processing: ' + error_path)

    # Parse info about the errors from the folder names
    error_sign = os.path.basename(error_path)
    error_type = error_sign.split('_')[0].split('=')[1]
    n_top = int(error_sign.split('_')[1].split('=')[1])
    res_sign = os.path.basename(os.path.dirname(error_path)).split('_')
    method = res_sign[0]
    dataset = res_sign[1]
    test_type = res_sign[2] if len(res_sign) > 3 else ''

    # Load dataset parameters
    dp = get_dataset_params(dataset, test_type=test_type)
    obj_ids = range(1, dp['obj_count'] + 1)
    scene_ids = range(1, dp['scene_count'] + 1)

    # Set threshold of correctness (might be different for each object)
    error_threshs = {}
    if error_type in ['add', 'adi']:
        # Relative to object diameter
        models_info = inout.load_yaml(dp['models_info_path'])
        for obj_id in obj_ids:
            obj_diameter = models_info[obj_id]['diameter']
            error_threshs[obj_id] = error_thresh_fact[error_type] * obj_diameter
    else:
        # The same threshold for all objects
        for obj_id in obj_ids:
            error_threshs[obj_id] = error_thresh[error_type]

    # Go through the test scenes and match estimated poses to GT poses
    matches = [] # Stores info about the matching estimate for each GT
    for scene_id in scene_ids:
        print('Matching: {}, {}, {}, {}, {}'.format(error_type, method, dataset,
                                                    test_type, scene_id))

        # Load GT poses
        gts = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

        # Load visibility fractions of the GT poses
        gt_visib_path = dp['scene_gt_stats_mpath'].format(scene_id, visib_delta)
        gt_visib = inout.load_yaml(gt_visib_path)

        # Load pre-calculated errors of the pose estimates
        scene_errs_path = errors_mpath.format(error_path=error_path,
                                              scene_id=scene_id)
        errs = inout.load_errors(scene_errs_path)

        # Organize the errors by image id and object id (for faster query)
        errs_org = {}
        for e in errs:
            errs_org.setdefault(e['im_id'], {}).\
                setdefault(e['obj_id'], []).append(e)

        # Matching
        for im_id, gts_im in gts.items():
            matches_im = [{
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': gt['obj_id'],
                'gt_id': gt_id,
                'est_id': -1,
                'score': -1,
                'error': -1,
                'error_norm': -1,
                'visib': gt_visib[im_id][gt_id],
                'valid': gt_visib[im_id][gt_id] >= visib_gt_min
            } for gt_id, gt in enumerate(gts_im)]

            # Mask of valid GT poses (i.e. GT poses with sufficient visibility)
            gt_valid_mask = [m['valid'] for m in matches_im]

            # Treat estimates of each object separately
            im_obj_ids = set([gt['obj_id'] for gt in gts_im])
            for obj_id in im_obj_ids:
                if im_id in errs_org.keys()\
                        and obj_id in errs_org[im_id].keys():

                    # Greedily match the estimated poses to the ground truth
                    # poses in the order of decreasing score
                    errs_im_obj = errs_org[im_id][obj_id]
                    ms = pose_matching.match_poses(errs_im_obj,
                                                   error_threshs[obj_id],
                                                   n_top, gt_valid_mask)
                    for m in ms:
                        g = matches_im[m['gt_id']]
                        g['est_id'] = m['est_id']
                        g['score'] = m['score']
                        g['error'] = m['error']
                        g['error_norm'] = m['error_norm']

            matches += matches_im

    # Calculation of performance scores
    #---------------------------------------------------------------------------
    # Count the number of visible object instances in each image
    insts = {i: {j: defaultdict(lambda: 0) for j in scene_ids} for i in obj_ids}
    for m in matches:
        if m['valid']:
            insts[m['obj_id']][m['scene_id']][m['im_id']] += 1

    # Count the number of targets = object instances to be found
    # (e.g. for 6D localization of a single instance of a single object, there
    # is either zero or one target in each image - there is just one even if
    # there are more instances of the object of interest)
    tars = 0 # Total number of targets
    obj_tars = {i: 0 for i in obj_ids} # Targets per object
    scene_tars = {i: 0 for i in scene_ids} # Targets per scene
    for obj_id, obj_insts in insts.items():
        for scene_id, scene_insts in obj_insts.items():

            # Count the number of targets in the current scene
            if n_top > 0:
                count = sum(np.minimum(n_top, scene_insts.values()))
            else: # 0 = all estimates, -1 = given by the number of GT poses
                count = sum(scene_insts.values())

            tars += count
            obj_tars[obj_id] += count
            scene_tars[scene_id] += count

    # Count the number of true positives
    tps = 0 # Total number of true positives
    obj_tps = {i: 0 for i in obj_ids} # True positives per object
    scene_tps = {i: 0 for i in scene_ids} # True positives per scene
    for m in matches:
        if m['valid'] and m['est_id'] != -1:
            tps += 1
            obj_tps[m['obj_id']] += 1
            scene_tps[m['scene_id']] += 1

    # Recall rates
    #---------------------------------------------------------------------------
    # Total recall
    total_recall = tps / float(tars)

    # Recall per object
    obj_recalls = {}
    for i in obj_ids:
        obj_recalls[i] = obj_tps[i] / float(obj_tars[i])
    mean_obj_recall = float(np.mean(obj_recalls.values()).squeeze())

    # Recall per scene
    scene_recalls = {}
    for i in scene_ids:
        scene_recalls[i] = scene_tps[i] / float(scene_tars[i])
    mean_scene_recall = float(np.mean(scene_recalls.values()).squeeze())

    #---------------------------------------------------------------------------
    scores = {
        'total_recall': total_recall,
        'obj_recalls': obj_recalls,
        'mean_obj_recall': mean_obj_recall,
        'scene_recalls': scene_recalls,
        'mean_scene_recall': mean_scene_recall
    }

    print('')
    print('GT count:           {:d}'.format(len(matches)))
    print('Target count:       {:d}'.format(tars))
    print('TP count:           {:d}'.format(tps))
    print('Total recall:       {:.4f}'.format(total_recall))
    print('Mean object recall: {:.4f}'.format(mean_obj_recall))
    print('Mean scene recall:  {:.4f}'.format(mean_scene_recall))
    # print('Object recalls:     {}'.format(str(obj_recalls)))
    # print('Scene recalls:      {}'.format(str(scene_recalls)))
    print('')

    # Evaluation signature
    eval_sign = ''
    if error_type in ['add', 'adi']:
        eval_sign = 'thf=' + str(error_thresh_fact[error_type])
    else:
        eval_sign = 'th=' + str(error_thresh[error_type])

    # Save scores
    print('Saving scores...')
    scores_path = scores_mpath.format(error_path=error_path,
                                      eval_sign=eval_sign)
    inout.save_yaml(scores_path, scores)

    # Save matches
    print('Saving matches...')
    matches_path = matches_mpath.format(error_path=error_path,
                                        eval_sign=eval_sign)
    inout.save_yaml(matches_path, matches)

    print('')
print('Done.')
