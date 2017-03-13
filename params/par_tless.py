# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

from pysixdb import inout

obj_count = 30
scene_count = 20

test_im_size = (640, 480)

dataset_name = 't-less'
base_path = '/local/datasets/tlod/t-less/t-less_v2/'
cam_params_path = base_path + 'camera{}.yml'

# Path masks
model_mpath = base_path + 'models_cad/obj_{:02d}.ply'
model_texture_mpath = None
obj_info_mpath = base_path + 'train_primesense/{:02d}/info.yml'
obj_gt_mpath = base_path + 'train_primesense/{:02d}/gt.yml'
train_rgb_mpath = base_path + 'train_primesense/{:02d}/rgb/{:04d}.png'
train_depth_mpath = base_path + 'train_primesense/{:02d}/depth/{:04d}.png'

scene_info_mpath = base_path + 'test_primesense/{:02d}/info.yml'
scene_gt_mpath = base_path + 'test_primesense/{:02d}/gt.yml'
test_rgb_mpath = base_path + 'test_primesense/{:02d}/rgb/{:04d}.png'
test_depth_mpath = base_path + 'test_primesense/{:02d}/depth/{:04d}.png'

cam = inout.load_cam_params(cam_params_path)
