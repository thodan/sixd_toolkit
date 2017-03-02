# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import glob
import cPickle as pickle
import yaml

base_path = '/local/datasets/tlod/imperial/tejani/orig/tejani_filtering/'
bad_path = base_path + 'gt_bad_annotation'
good_path = base_path + 'gt_good_annotation'
bad_out_path = base_path + 'bad_ids.p'
good_out_path = base_path + 'good_ids.p'
good_out_yaml_path = base_path + 'good_ids.yaml'

bad_ids = []
bad_fpaths = sorted(glob.glob(os.path.join(bad_path, '*.jpg')))
for fpath in bad_fpaths:
    scene_id, im_id = os.path.basename(fpath).split('.')[0].split('_')
    bad_ids.append((int(scene_id), int(im_id)))

good_ids = []
good_fpaths = sorted(glob.glob(os.path.join(good_path, '*.jpg')))
for fpath in good_fpaths:
    scene_id, im_id = os.path.basename(fpath).split('.')[0].split('_')
    good_ids.append((int(scene_id), int(im_id)))

with open(bad_out_path, 'w') as f:
    pickle.dump(bad_ids, f)

with open(good_out_path, 'w') as f:
    pickle.dump(good_ids, f)

# Save the good IDs into a YAML file
good_ids_scene = {}
for good_id in good_ids:
    orig_im_id = good_id[1] + 1
    if good_id[0] in good_ids_scene.keys():
        good_ids_scene[good_id[0]].append(orig_im_id)
    else:
        good_ids_scene[good_id[0]] = [orig_im_id]
with open(good_out_yaml_path, 'w') as f:
    yaml.dump(good_ids_scene, f)

good_counts = {}
for good_id in good_ids:
    if good_id[0] not in good_counts.keys():
        good_counts[good_id[0]] = 1
    else:
        good_counts[good_id[0]] += 1

bad_counts = {}
for bad_id in bad_ids:
    if bad_id[0] not in bad_counts.keys():
        bad_counts[bad_id[0]] = 1
    else:
        bad_counts[bad_id[0]] += 1

print('GOOD:')
for scene_id, count in good_counts.items():
    print('scene_id: {}, count: {}'.format(scene_id, count))

print('BAD:')
for scene_id, count in bad_counts.items():
    print('scene_id: {}, count: {}'.format(scene_id, count))
