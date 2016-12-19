# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates distances of objects from camera in test images of the selected
# dataset.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.abspath('..'))
from params import par_hinterstoisser as par

scene_ids = range(1, par.scene_count + 1)
obj_dists = []
for scene_id in scene_ids:
    print('Processing scene: ' + str(scene_id))
    with open(par.scene_gt_mpath.format(scene_id), 'r') as f:
        scene_gt = yaml.load(f, Loader=yaml.CLoader)

    for im_id, im_gts in scene_gt.items():
        for im_gt in im_gts:
            obj_dist = np.linalg.norm(im_gt['cam_t_m2c'])
            obj_dists.append(obj_dist)

print('Min obj distance: ' + str(min(obj_dists)))
print('Max obj distance: ' + str(max(obj_dists)))

plt.hist(obj_dists, bins=100)
