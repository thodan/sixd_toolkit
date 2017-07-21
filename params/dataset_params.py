# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys

sys.path.append(os.path.abspath('..'))
from pysixd import inout

def get_dataset_params(name, model_type='', train_type='', test_type='', cam_type=''):

    p = {'name': name, 'model_type': model_type,
         'train_type': train_type, 'test_type': test_type, 'cam_type': cam_type}

    # Folder with datasets
    common_base_path = '/local/datasets/sixd/'
    # common_base_path = '/datagrid/personal/hodanto2/datasets/sixd/'

    # Path to the T-LESS Toolkit (https://github.com/thodan/t-less_toolkit)
    tless_tk_path = '/home/tom/th_data/cmp/projects/t-less/t-less_toolkit/'
    # tless_tk_path = '/home.dokt/hodanto2/projects/t-less/t-less_toolkit/'

    if name == 'hinterstoisser':
        p['obj_count'] = 15
        p['scene_count'] = 15
        p['train_im_size'] = (640, 480)
        p['test_im_size'] = (640, 480)
        p['base_path'] = common_base_path + 'hinterstoisser/'
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None
        p['cam_params_path'] = p['base_path'] + 'camera.yml'

    elif name == 'tless':
        p['obj_count'] = 30
        p['scene_count'] = 20

        p['base_path'] = common_base_path + 't-less/t-less_v2/'
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None

        if p['model_type'] == '': p['model_type'] = 'cad'
        if p['train_type'] == '': p['train_type'] = 'primesense'
        if p['test_type'] == '': p['test_type'] = 'primesense'
        if p['cam_type'] == '': p['cam_type'] = 'primesense'

        p['cam_params_path'] = tless_tk_path + 'cam/camera_' + p['cam_type'] + '.yml'
        if p['test_type'] in ['primesense', 'kinect']:
            p['test_im_size'] = (720, 540)
        elif p['test_type'] == 'canon':
            p['test_im_size'] = (2560, 1920)

        if p['train_type'] in ['primesense', 'kinect']:
            p['train_im_size'] = (400, 400)
        elif p['train_type'] == 'canon':
            p['train_im_size'] = (1900, 1900)
        elif p['train_type'] == 'render_reconst':
            p['train_im_size'] = (1280, 1024)

    elif name == 'tudlight':
        p['obj_count'] = 3
        p['scene_count'] = 3
        p['train_im_size'] = (640, 480)
        p['test_im_size'] = (640, 480)
        p['base_path'] = common_base_path + 'tudlight/'
        p['im_id_pad'] = 5
        p['model_texture_mpath'] = None
        p['cam_params_path'] = p['base_path'] + 'camera.yml'

    elif name == 'rutgers':
        p['obj_count'] = 14
        p['scene_count'] = 14
        p['train_im_size'] = (640, 480)
        p['test_im_size'] = (640, 480)
        p['base_path'] = common_base_path + 'rutgers/'
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = p['base_path'] + 'models/obj_{:02d}.png'
        p['cam_params_path'] = p['base_path'] + 'camera.yml'

    elif name == 'tejani':
        p['obj_count'] = 6
        p['scene_count'] = 6
        p['train_im_size'] = (640, 480)
        p['test_im_size'] = (640, 480)
        p['base_path'] = common_base_path + 'tejani/'
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None
        p['cam_params_path'] = p['base_path'] + 'camera.yml'

    elif name == 'doumanoglou':
        p['obj_count'] = 2
        p['scene_count'] = 3
        p['train_im_size'] = (640, 480)
        p['test_im_size'] = (640, 480)
        p['base_path'] = common_base_path + 'doumanoglou/'
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None
        p['cam_params_path'] = p['base_path'] + 'camera.yml'

    else:
        print('Error: Unknown SIXD dataset.')
        exit(-1)

    models_dir = 'models' if p['model_type'] == '' else 'models_' + p['model_type']
    train_dir = 'train' if p['train_type'] == '' else 'train_' + p['train_type']
    test_dir = 'test' if p['test_type'] == '' else 'test_' + p['test_type']

    # Image ID format
    im_id_f = '{:' + str(p['im_id_pad']).zfill(2) + 'd}'

    # Paths and path masks
    p['model_mpath'] = p['base_path'] + models_dir + '/obj_{:02d}.ply'
    p['models_info_path'] = p['base_path'] + models_dir + '/models_info.yml'
    p['obj_info_mpath'] = p['base_path'] + train_dir + '/{:02d}/info.yml'
    p['obj_gt_mpath'] = p['base_path'] + train_dir + '/{:02d}/gt.yml'
    p['obj_gt_stats_mpath'] = p['base_path'] + train_dir + '_gt_stats/{:02d}_delta={}.yml'
    p['train_rgb_mpath'] = p['base_path'] + train_dir + '/{:02d}/rgb/' + im_id_f + '.png'
    p['train_depth_mpath'] = p['base_path'] + train_dir + '/{:02d}/depth/' + im_id_f + '.png'

    p['scene_info_mpath'] = p['base_path'] + test_dir + '/{:02d}/info.yml'
    p['scene_gt_mpath'] = p['base_path'] + test_dir + '/{:02d}/gt.yml'
    p['scene_gt_stats_mpath'] = p['base_path'] + test_dir + '_gt_stats/{:02d}_delta={}.yml'
    p['test_rgb_mpath'] = p['base_path'] + test_dir + '/{:02d}/rgb/' + im_id_f + '.png'
    p['test_depth_mpath'] = p['base_path'] + test_dir + '/{:02d}/depth/' + im_id_f + '.png'

    p['cam'] = inout.load_cam_params(p['cam_params_path'])

    return p
