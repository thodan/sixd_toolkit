
import os
import sys
import shutil
import cPickle as pickle
import numpy as np
import cv2

sys.path.append(os.path.abspath('..'))
from pysixdb import misc

stats_path = '/home/tom/th_data/cmp/projects/sixdb/sixdb_toolkit/output/eval_gt/rutgers_stats.p'
vis_gt_mpath = '/home/tom/th_data/cmp/projects/sixdb/sixdb_toolkit/output/vis_gt_poses_rutgers/{:02d}_{:04d}.jpg'
depth_diff_mpath = '/home/tom/th_data/cmp/projects/sixdb/sixdb_toolkit/output/eval_gt/rutgers/{:02d}_{:04d}_{:02d}_depth_diff.jpg'
rgb_mpath = '/local/datasets/tlod/rutgers/frank/test/{:02d}/rgb/{:04d}.png'
depth_mpath = '/local/datasets/tlod/rutgers/frank/test/{:02d}/depth/{:04d}.png'

out_path = '/home/tom/th_data/cmp/projects/sixdb/sixdb_toolkit/output/eval_gt_analysis'
misc.ensure_dir(out_path)
out_mbasename = '{:04d}_scene={}_im={}'
out_rgb_mpath = os.path.join(out_path, '{}_color.jpg')
out_depth_mpath = os.path.join(out_path, '{}_depth.jpg')
out_vis_gt_mpath = os.path.join(out_path, '{}_vis_gt.jpg')
out_depth_diff_mpath = os.path.join(out_path, '{}_depth_diff-mean={:.2f}_std={:.2f}_valid={:.2f}.jpg')

print('Loading stats...')
with open(stats_path, 'r') as f:
    stats = pickle.load(f)
print('stats loaded.')

# Analyse the depth differences
#-------------------------------------------------------------------------------
# Fraction of depth differences to be trimmed - to avoid erroneous depth
# measurements and to some extend also occlusion
trim_fract = 0.3

diff_all = []
diff_means = []
diff_stds = []
valid_fracs = []
scene_inds = {}
for stat_id, stat in enumerate(stats):
    min_keep_count = int(len(stat['diff']) * (1.0 - trim_fract))
    diff_trim = stat['diff'][np.argsort(np.abs(stat['diff']))[:min_keep_count]]

    diff_all += list(diff_trim)
    diff_means.append(np.mean(diff_trim))
    diff_stds.append(np.std(diff_trim))
    valid_fracs.append(stat['valid_frac'])

    if stat['scene_id'] in scene_inds.keys():
        scene_inds[stat['scene_id']].append(stat_id)
    else:
        scene_inds[stat['scene_id']] = [stat_id]

print('diff_all mean: ' + str(np.mean(diff_all)))
print('diff_all std: ' + str(np.std(diff_all)))

inds_order = np.argsort(np.abs(diff_means))[::-1]
# inds_order = np.argsort(np.abs(diff_stds))[::-1]
# inds_order = np.argsort(valid_fracs)

max_save = 200
# for i in range(max_save):
for i in range(len(inds_order)):
    ind = inds_order[i]
    stat = stats[ind]

    print('scene_id: {}, im_id: {}, gt_id: {}, mean diff: {}, std diff: {}'.format(
        stat['scene_id'], stat['im_id'], stat['gt_id'], diff_means[ind], diff_stds[ind]))

    if i < max_save:
        out_basename = out_mbasename.format(i, stat['scene_id'], stat['im_id'])
        out_rgb_fpath = out_rgb_mpath.format(out_basename)
        out_depth_fpath = out_depth_mpath.format(out_basename)
        out_vis_gt_fpath = out_vis_gt_mpath.format(out_basename)
        out_depth_diff_fpath = out_depth_diff_mpath.format(out_basename,
            diff_means[ind], diff_stds[ind], valid_fracs[ind])

        rgb = cv2.imread(rgb_mpath.format(stat['scene_id'], stat['im_id']))
        depth = cv2.imread(depth_mpath.format(stat['scene_id'], stat['im_id']),
                           cv2.IMREAD_UNCHANGED)
        depth = (255.0 / depth.max()) * depth.astype(np.float32)
        depth = depth.astype(np.uint8)

        cv2.imwrite(out_rgb_fpath, rgb)
        cv2.imwrite(out_depth_fpath, depth)
        shutil.copyfile(vis_gt_mpath.format(stat['scene_id'], stat['im_id']),
                        out_vis_gt_fpath)
        shutil.copyfile(depth_diff_mpath.format(
            stat['scene_id'], stat['im_id'], stat['gt_id']),
            out_depth_diff_fpath)

print('scene\tcount\tdiff_mean\tdiff_std\tvalid_perc')
for scene_id, inds in scene_inds.items():
    scene_diff_means = [diff_means[i] for i in inds]
    scene_diff_stds = [diff_stds[i] for i in inds]
    scene_valid_frac = [valid_fracs[i] for i in inds]
    print('{}\t\t{}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}'.format(scene_id, len(inds),
                                np.nanmean(scene_diff_means),
                                np.nanmean(scene_diff_stds),
                                np.nanmean(scene_valid_frac) * 100.0))
