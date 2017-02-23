# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

from pysixdb import inout

obj_count = 2
scene_count = 3

test_im_size = (640, 480)

dataset_name = 'doumanoglou'
base_path = '/local/datasets/tlod/imperial/doumanoglou_scenario_2/'
cam_params_path = base_path + 'camera.yml'

# Path masks
model_mpath = base_path + 'models/obj_{:02d}.ply'
model_texture_mpath = None
obj_info_mpath = base_path + 'train/obj_{:02d}/obj_info.yml'
train_rgb_mpath = base_path + 'train/obj_{:02d}/rgb/{:04d}.png'
train_depth_mpath = base_path + 'train/obj_{:02d}/depth/{:04d}.png'

scene_info_mpath = base_path + 'test/scene_{:02d}/scene_info.yml'
scene_gt_mpath = base_path + 'test/scene_{:02d}/scene_gt.yml'
test_rgb_mpath = base_path + 'test/scene_{:02d}/rgb/{:04d}.png'
test_depth_mpath = base_path + 'test/scene_{:02d}/depth/{:04d}.png'

cam = inout.load_cam_params(cam_params_path)
