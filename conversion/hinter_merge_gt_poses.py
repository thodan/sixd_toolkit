# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Copies a selected part of a dataset.

import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gt_poses_1_path = '/local/datasets/tlod/hinterstoisser/backup/scene_02_separated_gts/scene_gt.yml'
gt_poses_2_path = '/local/datasets/tlod/hinterstoisser/backup/scene_02_separated_gts/scene_gt_brachmann.yml'
gt_poses_out_path = '/local/datasets/tlod/hinterstoisser/test/02/scene_gt.yml'

with open(gt_poses_1_path, 'r') as f:
    gt_poses_1 = yaml.load(f, Loader=yaml.CLoader)
with open(gt_poses_2_path, 'r') as f:
    gt_poses_2 = yaml.load(f, Loader=yaml.CLoader)

assert(sorted(gt_poses_1.keys()) == (sorted(gt_poses_2.keys())))

gt_poses_out = {}
for im_id in sorted(gt_poses_1.keys()):
    gt_poses_out[im_id] = sorted(gt_poses_1[im_id] + gt_poses_2[im_id], key=lambda x: x['obj_id'])

def float_representer(dumper, value):
    text = '{0:.8f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
yaml.add_representer(float, float_representer)

# Store metadata
with open(gt_poses_out_path, 'w') as f:
    yaml.dump(gt_poses_out, f, width=10000)
