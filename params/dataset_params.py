# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

from os.path import normpath as np
from pysixd import inout

def get_dataset_params(name, model_type='', train_type='', test_type=''):

    p = {'name': name, 'model_type': model_type,
         'train_type': train_type, 'test_type': test_type}

    if name == 'hinterstoisser':
        p['obj_count'] = 3
        p['scene_count'] = 3
        p['test_im_size'] = (640, 480)
        p['base_path'] = np('/local/datasets/tlod/hinterstoisser/')
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None

    elif name == 'rutgers':
        p['obj_count'] = 14
        p['scene_count'] = 14
        p['test_im_size'] = (640, 480)
        p['base_path'] = np('/local/datasets/tlod/rutgers/')
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = p['base_path'] + 'models/obj_{:02d}.png'

    elif name == 'tejani':
        p['obj_count'] = 6
        p['scene_count'] = 6
        p['test_im_size'] = (640, 480)
        p['base_path'] = np('/local/datasets/tlod/imperial/tejani/')
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None

    elif name == 'tless':
        p['obj_count'] = 30
        p['scene_count'] = 20

        p['base_path'] = np('/local/datasets/tlod/t-less/t-less_v2/')
        p['im_id_pad'] = 4
        p['model_texture_mpath'] = None

        if p['model_type'] == '': p['model_type'] = 'cad'
        if p['train_type'] == '': p['train_type'] = 'primesense'
        if p['test_type'] == '': p['test_type'] = 'primesense'

        if p['test_type'] in ['primesense', 'kinect']:
            p['test_im_size'] = (720, 540)
        elif p['test_type'] == 'canon':
            p['test_im_size'] = (2560, 1920)

    elif name == 'tudlight':
        p['obj_count'] = 3
        p['scene_count'] = 3
        p['test_im_size'] = (640, 480)
        p['base_path'] = np('/local/datasets/tlod/dresden/tudlight/')
        p['im_id_pad'] = 5
        p['model_texture_mpath'] = None

        if p['train_type'] == '': p['train_type'] = 'real'

    models_dir = 'models' if p['model_type'] == '' else 'models_' + p['model_type']
    train_dir = 'train' if p['train_type'] == '' else 'train_' + p['train_type']
    test_dir = 'test' if p['test_type'] == '' else 'test_' + p['test_type']

    # Image ID format
    im_id_f = '{:' + p['im_id_pad'].zfill(2) + 'd}'

    # Path masks
    p['model_mpath'] = np(p['base_path'] + models_dir + '/obj_{:02d}.ply')
    p['obj_info_mpath'] = np(p['base_path'] + train_dir + '/{:02d}/info.yml')
    p['obj_gt_mpath'] = np(p['base_path'] + train_dir + '/{:02d}/gt.yml')
    p['train_rgb_mpath'] = np(p['base_path'] + train_dir + '/{:02d}/rgb/' + im_id_f + '.png')
    p['train_depth_mpath'] = np(p['base_path'] + train_dir + '/{:02d}/depth/' + im_id_f + '.png')

    p['scene_info_mpath'] = np(p['base_path'] + test_dir + '/{:02d}/info.yml')
    p['scene_gt_mpath'] = np(p['base_path'] + test_dir + '/{:02d}/gt.yml')
    p['test_rgb_mpath'] = np(p['base_path'] + test_dir + '/{:02d}/rgb/' + im_id_f + '.png')
    p['test_depth_mpath'] = np(p['base_path'] + test_dir + '/{:02d}/depth/' + im_id_f + '.png')

    p['cam_params_path'] = np(p['base_path'] + 'camera.yml')
    p['cam'] = inout.load_cam_params(p['cam_params_path'])

    return p
