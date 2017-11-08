# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Note: The last RGB-D image of the Benchvise sequence of the Hinterstoisser's
# dataset was removed, because Brachmann et al. do not provide the extended
# ground truth poses for it.

import os
import sys
import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, transform, renderer
import hinter_flip

from params.dataset_params import get_dataset_params
par = get_dataset_params('hinterstoisser')

base_path = '/local/datasets/tlod/hinterstoisser/'
rgb_mpath = base_path + 'test/02/rgb/{:04d}.png'
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
pose_mpath = '/local/datasets/tlod/dresden/occlusion/poses/{}/info_{:05d}.txt'
scene_gt_path = base_path + 'test/02/scene_gt_brachmann.yml'

obj_names_id_map = {'Ape': 1, 'Can': 5, 'Cat': 6, 'Driller': 8, 'Duck': 9,
                    'Eggbox': 10, 'Glue': 11, 'Holepuncher': 12}

def load_gt_pose_brachmann(path):
    R = []
    t = []
    rotation_sec = False
    center_sec = False
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            if 'rotation:' in line:
                rotation_sec = True
            elif rotation_sec:
                R += line.split(' ')
                if len(R) == 9:
                    rotation_sec = False
            elif 'center:' in line:
                center_sec = True
            elif center_sec:
                t = line.split(' ')
                center_sec = False

    assert((len(R) == 0 and len(t) == 0) or
           (len(R) == 9 and len(t) == 3))

    if len(R) == 0:
        pose = {'R': np.array([]), 't': np.array([])}
    else:
        pose = {'R': np.array(map(float, R)).reshape((3, 3)),
                't': np.array(map(float, t)).reshape((3, 1))}

        # Flip Y and Z axis (OpenGL -> OpenCV coordinate system)
        yz_flip = np.eye(3, dtype=np.float32)
        yz_flip[0, 0], yz_flip[1, 1], yz_flip[2, 2] = 1, -1, -1
        pose['R'] = yz_flip.dot(pose['R'])
        pose['t'] = yz_flip.dot(pose['t'])
    return pose

# Get list of image IDs
rgb_fpaths = sorted(glob.glob(os.path.dirname(pose_mpath.format('Ape', 0)) + '/*.txt'))
im_ids = sorted([int(e.split('info_')[1].split('.txt')[0]) for e in rgb_fpaths])

scene_gt = {}
for obj_name in sorted(obj_names_id_map.keys()):

    # Load object model
    obj_id = obj_names_id_map[obj_name]
    model = inout.load_ply(model_mpath.format(obj_id))

    # Transformation which was applied to the object models (its inverse will
    # be applied to the GT poses):
    # 1) Translate the bounding box center to the origin - Brachmann et al.
    # already translated the bounding box to the center
    # 2) Rotate around Y axis by pi + flip for some objects
    R_model = transform.rotation_matrix(math.pi, [0, 1, 0])[:3, :3]

    # Extra rotation around Z axis by pi for some models
    if hinter_flip.obj_flip_z[obj_id]:
        R_z = transform.rotation_matrix(math.pi, [0, 0, 1])[:3, :3]
        R_model = R_z.dot(R_model)

    # The ground truth poses of Brachmann et al. are related to a different
    # model coordinate system - to get the original Hinterstoisser's orientation
    # of the objects, we need to rotate by pi/2 around X and by pi/2 around Z
    R_z_90 = transform.rotation_matrix(-math.pi * 0.5, [0, 0, 1])[:3, :3]
    R_x_90 = transform.rotation_matrix(-math.pi * 0.5, [1, 0, 0])[:3, :3]
    R_conv = np.linalg.inv(R_model.dot(R_z_90.dot(R_x_90)))

    for im_id in im_ids:
        if im_id % 10 == 0:
            print('obj,view: ' + obj_name + ',' + str(im_id))

        # Load the GT pose
        pose = load_gt_pose_brachmann(pose_mpath.format(obj_name, im_id))
        if pose['R'].size != 0 and pose['t'].size != 0:

            # Transfom the GT pose
            R_m2c = pose['R'].dot(R_conv)
            t_m2c = pose['t'] * 1000 # from [m] to [mm]

            # Get 2D bounding box of the object model at the ground truth pose
            obj_bb = misc.calc_pose_2d_bbox(model, par['cam']['im_size'],
                                            par['cam']['K'], R_m2c, t_m2c)

            # Visualization
            if False:
                rgb = inout.load_im(rgb_mpath.format(im_id, im_id))
                ren_rgb = renderer.render(model, par['cam']['im_size'],
                                          par['cam']['K'], R_m2c, t_m2c, mode='rgb')
                vis_rgb = 0.4 * rgb.astype(np.float32) +\
                          0.6 * ren_rgb.astype(np.float32)
                vis_rgb = vis_rgb.astype(np.uint8)
                vis_rgb = misc.draw_rect(vis_rgb, obj_bb)
                plt.imshow(vis_rgb)
                plt.show()

            scene_gt.setdefault(im_id, []).append(
                {
                 'obj_id': obj_id,
                 'cam_R_m2c': R_m2c.flatten().tolist(),
                 'cam_t_m2c': t_m2c.flatten().tolist(),
                 'obj_bb': [int(x) for x in obj_bb]
                }
            )

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    yaml.add_representer(float, float_representer)

    # Store ground truth poses
    with open(scene_gt_path, 'w') as f:
        yaml.dump(scene_gt, f, width=10000)
