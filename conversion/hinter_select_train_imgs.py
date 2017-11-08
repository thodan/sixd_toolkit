# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Copies a selected part of a dataset.

import os
import sys
import shutil
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import misc

obj_ids = range(1, 16)
# obj_ids = range(14, 16)

select_im_ids = range(1313)

base_path = '/local/datasets/tlod/hinterstoisser/'

in_obj_info_mpath = base_path + 'train_400-500/{:02d}/info.yml'
in_obj_gt_mpath = base_path + 'train_400-500/{:02d}/gt.yml'
in_rgb_mpath = base_path + 'train_400-500/{:02d}/rgb/{:04d}.png'
in_depth_mpath = base_path + 'train_400-500/{:02d}/depth/{:04d}.png'

out_obj_info_mpath = base_path + 'train/{:02d}/info.yml'
out_obj_gt_mpath = base_path + 'train/{:02d}/gt.yml'
out_rgb_mpath = base_path + 'train/{:02d}/rgb/{:04d}.png'
out_depth_mpath = base_path + 'train/{:02d}/depth/{:04d}.png'

for obj_id in obj_ids:
    # Prepare folders
    misc.ensure_dir(os.path.dirname(out_rgb_mpath.format(obj_id, 0)))
    misc.ensure_dir(os.path.dirname(out_depth_mpath.format(obj_id, 0)))

    # Load object info and gt
    with open(in_obj_info_mpath.format(obj_id), 'r') as f:
        in_obj_info = yaml.load(f, Loader=yaml.CLoader)
    with open(in_obj_gt_mpath.format(obj_id), 'r') as f:
        in_obj_gt = yaml.load(f, Loader=yaml.CLoader)

    out_obj_info = {}
    out_obj_gt = {}
    for im_id in sorted(in_obj_info.keys()):
        if im_id not in select_im_ids:
            continue

        shutil.copyfile(in_rgb_mpath.format(obj_id, im_id),
                        out_rgb_mpath.format(obj_id, im_id))

        shutil.copyfile(in_depth_mpath.format(obj_id, im_id),
                        out_depth_mpath.format(obj_id, im_id))

        out_obj_gt[im_id] = in_obj_gt[im_id]
        out_obj_info[im_id] = in_obj_info[im_id]
        if 'sphere_radius' in out_obj_info[im_id].keys():
            del out_obj_info[im_id]['sphere_radius']

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    yaml.add_representer(float, float_representer)

    # Store metadata
    with open(out_obj_info_mpath.format(obj_id), 'w') as f:
        yaml.dump(out_obj_info, f, width=10000)
    with open(out_obj_gt_mpath.format(obj_id), 'w') as f:
        yaml.dump(out_obj_gt, f, width=10000)
