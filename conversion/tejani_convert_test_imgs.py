# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys
import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer

from params.dataset_params import get_dataset_params
par = get_dataset_params('tejani')

base_path = '/local/datasets/tlod/imperial/tejani/'

rgb_in_mpath = base_path + 'orig/test/scene_{:02d}/RGB/img_{:03d}.png'
depth_in_mpath = base_path + 'orig/test/scene_{:02d}/Depth/img_{:03d}.png'
pose_mpath = base_path + 'orig/annotation_new_models/scene_{:02d}/poses{}.txt'
model_mpath = base_path + 'models/obj_{:02d}.ply'
bbox_cens_path = base_path + 'bbox_cens.yml'

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'
rgb_out_mpath = base_path + 'test/{:02d}/rgb/{:04d}.png'
depth_out_mpath = base_path + 'test/{:02d}/depth/{:04d}.png'

# scene_ids = range(1, 7)
scene_ids = range(3, 7)

def load_tejani_poses(path):
    '''
    Loads poses from a text file as used by Tejani.
    '''
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        mat_count = int(lines[0])
        poses = []
        for mat_id in range(mat_count):
            mat = np.zeros((4, 4), np.float32)
            for y in range(4):
                # Normalize the line (sometimes comma and sometimes
                # whitespace is used as a separator)
                line = lines[1 + 4 * mat_id + y].lstrip().rstrip().replace(',', ' ')
                mat[y, :] = np.array(map(float, line.split()))
            poses.append({'R': mat[:3, :3], 't': mat[:3, 3].reshape((3, 1))})
    return poses

with open(bbox_cens_path, 'r') as f:
    bbox_cens = np.array(yaml.load(f))

for scene_id in scene_ids:
    scene_info = {}
    scene_gt = {}

    # Prepare folders
    misc.ensure_dir(os.path.dirname(rgb_out_mpath.format(scene_id, 0)))
    misc.ensure_dir(os.path.dirname(depth_out_mpath.format(scene_id, 0)))

    # Get list of image IDs - consider only images for which the fixed GT
    # is available
    poses_fpaths = glob.glob(os.path.join(
        os.path.dirname(pose_mpath.format(scene_id, 0)), '*'))
    im_ids = sorted([int(e.split('poses')[1].split('.txt')[0]) for e in poses_fpaths])

    # Load object model
    obj_id = scene_id  # The object id is the same as scene id for this dataset
    model = inout.load_ply(model_mpath.format(obj_id))

    # Transformation which was applied to the object models (its inverse will
    # be applied to the GT poses):
    # 1) Translate the bounding box center to the origin
    t_model = bbox_cens[obj_id - 1, :].reshape((3, 1))

    im_id_out = 0
    for im_id in im_ids:
        # if im_id % 10 == 0:
        print('scene,view: ' + str(scene_id) + ',' + str(im_id))

        # Load the RGB and depth image
        rgb = inout.load_im(rgb_in_mpath.format(scene_id, im_id))
        depth = inout.load_depth(depth_in_mpath.format(scene_id, im_id))

        depth *= 10.0  # Convert depth map to [100um]

        # Save the RGB and depth image
        inout.save_im(rgb_out_mpath.format(scene_id, im_id_out), rgb)
        inout.save_depth(depth_out_mpath.format(scene_id, im_id_out), depth)

        scene_info[im_id_out] = {
            'cam_K': par['cam']['K'].flatten().tolist()
        }

        # Process the GT poses
        poses = load_tejani_poses(pose_mpath.format(scene_id, im_id))
        scene_gt[im_id_out] = []
        for pose in poses:
            R_m2c = pose['R']
            t_m2c = pose['t']

            t_m2c *= 1000.0 # Convert to [mm]

            # Transfom the GT pose (to compensate transformation of the models)
            t_m2c = t_m2c + R_m2c.dot(t_model)

            # Get 2D bounding box of the object model at the ground truth pose
            obj_bb = misc.calc_pose_2d_bbox(model, par['cam']['im_size'], par['cam']['K'], R_m2c, t_m2c)

            # Visualisation
            if False:
                ren_rgb = renderer.render(model, par['cam']['im_size'], par['cam']['K'],
                                          R_m2c, t_m2c, mode='rgb')
                vis_rgb = 0.4 * rgb.astype(np.float32) + 0.6 * ren_rgb.astype(np.float32)
                vis_rgb = vis_rgb.astype(np.uint8)
                vis_rgb = misc.draw_rect(vis_rgb, obj_bb)
                plt.imshow(vis_rgb)
                plt.show()

            scene_gt[im_id_out].append(
                {
                 'obj_id': obj_id,
                 'cam_R_m2c': R_m2c.flatten().tolist(),
                 'cam_t_m2c': t_m2c.flatten().tolist(),
                 'obj_bb': [int(x) for x in obj_bb]
                }
            )

        im_id_out += 1

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    yaml.add_representer(float, float_representer)

    # Store metadata
    with open(scene_info_mpath.format(scene_id), 'w') as f:
        yaml.dump(scene_info, f, width=10000)

    # Store ground truth poses
    with open(scene_gt_mpath.format(scene_id), 'w') as f:
        yaml.dump(scene_gt, f, width=10000)
