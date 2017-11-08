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
par = get_dataset_params('doumanoglou')

rgb_in_mpath = par['base_path'] + 'orig/test/{:02d}/rgb{}.png'
depth_in_mpath = par['base_path'] + 'orig/test/{:02d}/depth{}.png'
cam_pose_mpath = par['base_path'] + 'orig/test/{:02d}/camera_pose_{}.txt'
obj_pose_mpath = par['base_path'] + 'orig/test/{:02d}/{}{}_{}.txt'
model_mpath = par['base_path'] + 'models/obj_{:02d}.ply'
bbox_cens_path = par['base_path'] + 'bbox_cens.yml'

scene_info_mpath = par['base_path'] + 'test/{:02d}/info.yml'
scene_gt_mpath = par['base_path'] + 'test/{:02d}/gt.yml'
rgb_out_mpath = par['base_path'] + 'test/{:02d}/rgb/{:04d}.png'
depth_out_mpath = par['base_path'] + 'test/{:02d}/depth/{:04d}.png'

# IDs and counts of objects included in the scenes
scenes = {1: {1: 15}, 2: {2: 5}, 3: {1: 9, 2: 4}}
# scenes = {2: {2: 5}, 3: {1: 9, 2: 4}}
obj_names = {1: 'coffee_cup', 2: 'juice'}

def load_doumanoglou_pose(path, convert):
    '''
    Loads a pose from a text file as used by Doumanoglou.
    '''
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        mat = np.zeros((4, 4), np.float32)
        for i in range(4):
            mat[i, :] = np.array(map(float, lines[i].split()))
        R = mat[:3, :3]
        t = mat[:3, 3].reshape((3, 1))

        # Flip Y and Z axis (OpenGL -> OpenCV coordinate system)
        if convert:
            yz_flip = np.eye(3, dtype=np.float32)
            yz_flip[0, 0], yz_flip[1, 1], yz_flip[2, 2] = 1, -1, -1
            R = yz_flip.dot(R)
            t = yz_flip.dot(t)

    return R, t

def calc_center_of_mass(model):
    '''
    Center of mass of a model as calculated by Imperial College.
    '''
    com = np.zeros((1, 3))
    total_area = 0
    for face in model['faces']:
        assert(face.size == 3)
        pts = [model['pts'][i] for i in face]
        face_cen = (pts[0] + pts[1] + pts[2]) / 3.0
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])

        # Calculate triangle area
        s = (a + b + c) / 2.0
        face_area = (s * (s - a) * (s - b) * (s - c)) ** 0.5

        com += face_cen * face_area
        total_area += face_area
    com /= total_area
    return com

with open(bbox_cens_path, 'r') as f:
    bbox_cens = np.array(yaml.load(f))

# Transformations from the model coordinate system used for models in scenario 2
# of the Doumanoglou's dataset to the model coordinate system of the corresponding
# models from the Tejani's dataset.
# These transformations were provided by Caner Sahin from Imperial College.
trans_dou2tejani = {
    1: { # coffee
        'R': np.array(
            [[1.0000, 0.0000, 0.0000],
             [0.0000, -1.0000, 0.0000],
             [0.0000, 0.0000, -1.0000]]
        ),
        't': np.array([[0.0070, 0.0049, -0.0702]]).T * 1000
    },
    2: { # juice
        'R': np.array(
            [[1.0000, 0.0000, 0.0000],
             [0.0000, -1.0000, 0.0000],
             [0.0000, 0.0000, -1.0000]]
        ),
        't': np.array([[0.0023, 0.0100, -0.1089]]).T * 1000
    }
}

# Transformations from Tejani to Doumanoglou
trans_tejani2dou = {
    1: {
        'R': np.linalg.inv(trans_dou2tejani[1]['R']),
        't': -np.linalg.inv(trans_dou2tejani[1]['R']).dot(trans_dou2tejani[1]['t'])
    },
    2: {
        'R': np.linalg.inv(trans_dou2tejani[2]['R']),
        't': -np.linalg.inv(trans_dou2tejani[2]['R']).dot(trans_dou2tejani[2]['t'])
    }
}

# Test of the transformations
# p_in = '/local/datasets/tlod/imperial/doumanoglou_scenario_2/models/old/obj_02.ply'
# p_out = '/local/datasets/tlod/imperial/doumanoglou_scenario_2/models/old/obj_02_tejani.ply'
# m = inout.load_ply(p_in)
# # yz_flip = np.eye(3, dtype=np.float32)
# # yz_flip[0, 0], yz_flip[1, 1], yz_flip[2, 2] = 1, -1, -1
# # m['pts'] *= 1000
# # m['pts'] = yz_flip.dot(m['pts'].T).T
# # m_com = calc_center_of_mass(m)
# # m['pts'] -= m_com
# # m['pts'] -= bbox_cens[0, :].reshape((1, 3))
# m['pts'] = (trans_dou2tejani[2]['R'].dot(m['pts'].T) + trans_dou2tejani[2]['t'] / 1000.0).T
# inout.save_ply(p_out, m['pts'], m['colors'], faces=m['faces'])
# exit(-1)

for scene_id in sorted(scenes.keys()):
    scene_info = {}
    scene_gt = {}

    # Prepare folders
    misc.ensure_dir(os.path.dirname(rgb_out_mpath.format(scene_id, 0)))
    misc.ensure_dir(os.path.dirname(depth_out_mpath.format(scene_id, 0)))

    # Get list of image IDs
    poses_fpaths = glob.glob(cam_pose_mpath.format(scene_id, '*'))
    im_ids = sorted([int(e.split('camera_pose_')[1].split('.txt')[0]) for e in poses_fpaths])

    # Load object models
    models = {}
    #models_com = {}
    ts_model = {}
    for obj_id in scenes[scene_id].keys():
        models[obj_id] = inout.load_ply(model_mpath.format(obj_id))
        # models[obj_id]['pts'] *= 1000.0
        #models_com[obj_id] = calc_center_of_mass(models[obj_id])

        # Transformation which was applied to the object models (its inverse will
        # be applied to the GT poses):
        ts_model[obj_id] = bbox_cens[obj_id - 1, :].reshape((3, 1))

    im_id_out = 0
    for im_id in im_ids:
        # if im_id % 10 == 0:
        print('scene, view: ' + str(scene_id) + ', ' + str(im_id))

        # Load the RGB and depth image
        rgb = inout.load_im(rgb_in_mpath.format(scene_id, im_id))
        depth = inout.load_depth(depth_in_mpath.format(scene_id, im_id))

        #depth *= 10.0  # Convert depth map to [100um]

        # Save the RGB and depth image
        inout.save_im(rgb_out_mpath.format(scene_id, im_id_out), rgb)
        inout.save_depth(depth_out_mpath.format(scene_id, im_id_out), depth)

        # Load the camera pose
        cam_R, cam_t = load_doumanoglou_pose(cam_pose_mpath.format(scene_id, im_id), False)

        scene_info[im_id_out] = {
            'cam_K': par['cam']['K'],
            'cam_R_w2c': cam_R,
            'cam_t_w2c': 1000.0 * cam_t # [mm]
        }

        # Process the GT poses
        scene_gt[im_id_out] = []
        for obj_id in sorted(scenes[scene_id].keys()):
            obj_count = scenes[scene_id][obj_id]
            for i in range(obj_count):
                R, t = load_doumanoglou_pose(obj_pose_mpath.format(
                    scene_id, obj_names[obj_id], i + 1, im_id), True)

                # The GT poses are expressed w.r.t. the models provided by
                # Doumanoglou, but we want them to be expressed w.r.t. the model
                # coordinate system used by Tejani.
                R_m2c = R.dot(trans_tejani2dou[obj_id]['R'])
                t_m2c = 1000.0 * t + R_m2c.dot(ts_model[obj_id]) +\
                        R.dot(trans_tejani2dou[obj_id]['t'])

                # Get 2D bounding box of the object model at the ground truth pose
                obj_bb = misc.calc_pose_2d_bbox(models[obj_id], par['cam']['im_size'],
                                                par['cam']['K'], R_m2c, t_m2c)

                # Visualisation
                if False:
                    print(R_m2c)
                    print(t_m2c)

                    ren_rgb = renderer.render(models[obj_id], par['cam']['im_size'], par['cam']['K'],
                                              R_m2c, t_m2c, mode='rgb')
                    vis_rgb = 0.4 * rgb.astype(np.float32) + 0.6 * ren_rgb.astype(np.float32)
                    vis_rgb = vis_rgb.astype(np.uint8)
                    vis_rgb = misc.draw_rect(vis_rgb, obj_bb)
                    plt.imshow(vis_rgb)
                    plt.show()

                scene_gt[im_id_out].append(
                    {
                     'obj_id': obj_id,
                     'cam_R_m2c': R_m2c,
                     'cam_t_m2c': t_m2c,
                     'obj_bb': obj_bb
                    }
                )

        im_id_out += 1

    inout.save_info(scene_info_mpath.format(scene_id), scene_info)
    inout.save_gt(scene_gt_mpath.format(scene_id), scene_gt)
