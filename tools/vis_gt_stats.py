# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Plots statistics of the ground truth poses.

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from pysixd import inout
from params.dataset_params import get_dataset_params

# dataset = 'hinterstoisser'
dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'

delta = 15 # Tolerance used in the visibility test [mm]

# Load dataset parameters
dp = get_dataset_params(dataset)
obj_ids = range(1, dp['obj_count'] + 1)
scene_ids = range(1, dp['scene_count'] + 1)

# Load the GT statistics
gt_stats = []
for scene_id in scene_ids:
    print('Loading GT stats: {}, {}'.format(dataset, scene_id))
    gts = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
    gt_stats_curr = inout.load_yaml(
        dp['scene_gt_stats_mpath'].format(scene_id, delta))
    for im_id, gt_stats_im in gt_stats_curr.items():
        for gt_id, p in enumerate(gt_stats_im):
            p['scene_id'] = scene_id
            p['im_id'] = im_id
            p['gt_id'] = gt_id
            p['obj_id'] = gts[im_id][gt_id]['obj_id']
            gt_stats.append(p)

print('GT count: {}'.format(len(gt_stats)))

# Collect the data
px_count_all = [p['px_count_all'] for p in gt_stats]
px_count_valid = [p['px_count_valid'] for p in gt_stats]
px_count_visib = [p['px_count_visib'] for p in gt_stats]
visib_fract = [p['visib_fract'] for p in gt_stats]
bbox_all_x = [p['bbox_all'][0] for p in gt_stats]
bbox_all_y = [p['bbox_all'][1] for p in gt_stats]
bbox_all_w = [p['bbox_all'][2] for p in gt_stats]
bbox_all_h = [p['bbox_all'][3] for p in gt_stats]
bbox_visib_x = [p['bbox_visib'][0] for p in gt_stats]
bbox_visib_y = [p['bbox_visib'][1] for p in gt_stats]
bbox_visib_w = [p['bbox_visib'][2] for p in gt_stats]
bbox_visib_h = [p['bbox_visib'][3] for p in gt_stats]

f, axs = plt.subplots(2, 2)
f.canvas.set_window_title(dataset)

axs[0, 0].hist([px_count_all, px_count_valid, px_count_visib],
            bins=20, range=(min(px_count_visib), max(px_count_all)))
axs[0, 0].legend([
    'All object mask pixels',
    'Valid object mask pixels',
    'Visible object mask pixels'
])

axs[0, 1].hist(visib_fract, bins=50, range=(0.0, 1.0))
axs[0, 1].set_xlabel('Visible fraction')

axs[1, 0].hist([bbox_all_x, bbox_all_y, bbox_visib_x, bbox_visib_y], bins=20)
axs[1, 0].legend([
    'Bbox all - x',
    'Bbox all - y',
    'Bbox visib - x',
    'Bbox visib - y'
])

axs[1, 1].hist([bbox_all_w, bbox_all_h, bbox_visib_w, bbox_visib_h], bins=20)
axs[1, 1].legend([
    'Bbox all - width',
    'Bbox all - height',
    'Bbox visib - width',
    'Bbox visib - height'
])

f.tight_layout()
plt.show()
