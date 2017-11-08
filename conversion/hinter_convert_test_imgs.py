# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys
import glob
import math
import struct

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, transform, renderer
import hinter_flip

from params.dataset_params import get_dataset_params
par = get_dataset_params('hinterstoisser')

base_path = '/local/datasets/tlod/hinterstoisser/'

rgb_in_mpath = base_path + 'orig/test/scene_{:02d}/color{}.jpg'
depth_in_mpath = base_path + 'orig/test/scene_{:02d}/depth{}.dpt'
rot_mpath = base_path + 'orig/test/scene_{:02d}/rot{}.rot' # File with GT rotation
tra_mpath = base_path + 'orig/test/scene_{:02d}/tra{}.tra' # File with GT translation
model_mpath = base_path + 'models/obj_{:02d}.ply' # Already transformed
bbox_cens_path = 'output/bbox_cens.yml'

scene_info_mpath = base_path + 'test/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test/{:02d}/gt.yml'
rgb_out_mpath = base_path + 'test/{:02d}/rgb/{:04d}.png'
depth_out_mpath = base_path + 'test/{:02d}/depth/{:04d}.png'

scene_ids = range(1, 16)

INT_BYTES = 4
USHORT_BYTES = 2

def load_hinter_depth(path):
    '''
    Loads depth image from the '.dpt' format used by Hinterstoisser.
    '''
    f = open(path, 'rb')
    h = struct.unpack('i', f.read(INT_BYTES))[0]
    w = struct.unpack('i', f.read(INT_BYTES))[0]
    depth_map = []
    for i in range(h):
        depth_map.append(struct.unpack(w*'H', f.read(w * USHORT_BYTES)))
    # return np.array(depth_map, np.uint16)
    return np.array(depth_map, np.float32)

def load_hinter_mat(path):
    '''
    Loads matrix from a text file as used by Hinterstoisser to save rotation
    and translation vector.
    '''
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        mat_size = map(int, lines[0].split(' '))
        mat = np.zeros((mat_size[1], mat_size[0]), np.float32)
        for y in range(mat_size[1]):
            line_elems = map(float, [e for e in lines[y + 1].split(' ') if e != ''])
            for x in range(mat_size[0]):
                mat[y, x] = line_elems[x]
    return mat

with open(bbox_cens_path, 'r') as f:
    bbox_cens = np.array(yaml.load(f))

for scene_id in scene_ids:
    scene_info = {}
    scene_gt = {}

    # Prepare folders
    misc.ensure_dir(os.path.dirname(rgb_out_mpath.format(scene_id, 0)))
    misc.ensure_dir(os.path.dirname(depth_out_mpath.format(scene_id, 0)))

    # Get list of image IDs
    color_fpaths = glob.glob(rgb_in_mpath.format(scene_id, '*'))
    im_ids = sorted([int(e.split('color')[1].split('.jpg')[0]) for e in color_fpaths])

    # Load object model
    obj_id = scene_id  # The object id is the same as scene id for this dataset
    model = inout.load_ply(model_mpath.format(obj_id))

    # Transformation which was applied to the object models (its inverse will
    # be applied to the GT poses):
    # 1) Translate the bounding box center to the origin
    # 2) Rotate around Y axis by pi + flip for some objects
    t_model = bbox_cens[obj_id - 1, :].reshape((3, 1))
    R_model = transform.rotation_matrix(math.pi, [0, 1, 0])[:3, :3]

    # Extra rotation around Z axis by pi for some models
    if hinter_flip.obj_flip_z[obj_id]:
        R_z = transform.rotation_matrix(math.pi, [0, 0, 1])[:3, :3]
        R_model = R_z.dot(R_model)

    R_model_inv = np.linalg.inv(R_model)

    for im_id in im_ids:
        if im_id % 10 == 0:
            print('scene,view: ' + str(scene_id) + ',' + str(im_id))

        # Load the RGB and depth image
        rgb = inout.load_im(rgb_in_mpath.format(scene_id, im_id))
        depth = load_hinter_depth(depth_in_mpath.format(scene_id, im_id))

        depth *= 10.0  # Convert depth map to [100um]

        # Save the RGB and depth image
        inout.save_im(rgb_out_mpath.format(scene_id, im_id), rgb)
        inout.save_depth(depth_out_mpath.format(scene_id, im_id), depth)

        # Load the GT pose
        R_m2c = load_hinter_mat(rot_mpath.format(scene_id, im_id))
        t_m2c = load_hinter_mat(tra_mpath.format(scene_id, im_id))
        t_m2c *= 10 # Convert to [mm]

        # Transfom the GT pose (to compensate transformation of the models)
        R_m2c = R_m2c.dot(R_model_inv)
        t_m2c = t_m2c + R_m2c.dot(R_model.dot(t_model))

        # Get 2D bounding box of the object model at the ground truth pose
        obj_bb = misc.calc_pose_2d_bbox(model, par['cam']['im_size'],
                                        par['cam']['K'], R_m2c, t_m2c)

        # Visualisation
        if False:
            ren_rgb = renderer.render(model, par['cam']['im_size'],
                                      par['cam']['K'], R_m2c, t_m2c, mode='rgb')
            vis_rgb = 0.4 * rgb.astype(np.float32) +\
                      0.6 * ren_rgb.astype(np.float32)
            vis_rgb = vis_rgb.astype(np.uint8)
            vis_rgb = misc.draw_rect(vis_rgb, obj_bb)
            plt.imshow(vis_rgb)
            plt.show()

        scene_gt[im_id] = [
            {
             'obj_id': obj_id,
             'cam_R_m2c': R_m2c.flatten().tolist(),
             'cam_t_m2c': t_m2c.flatten().tolist(),
             'obj_bb': [int(x) for x in obj_bb]
            }
        ]

        scene_info[im_id] = {
            'cam_K': par['cam']['K'].flatten().tolist()
        }

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
