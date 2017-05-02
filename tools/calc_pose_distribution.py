# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates distances of objects from camera in test images of the selected
# dataset.

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from pysixd import inout

from params.dataset_params import get_dataset_params
# par = get_dataset_params('hinterstoisser')
# par = get_dataset_params('tejani')
par = get_dataset_params('rutgers')
# par = get_dataset_params('tudlight')

scene_ids = range(1, par['scene_count'] + 1)
obj_dists = []
azimuths = []
elevs = []
ims_count = 0
for scene_id in scene_ids:
    print('Processing scene: ' + str(scene_id))
    scene_gt = inout.load_gt(par['scene_gt_mpath'].format(scene_id))
    ims_count += len(scene_gt)

    for im_id, im_gts in scene_gt.items():
        for im_gt in im_gts:
            if scene_id == 2 and im_gt['obj_id'] != 2:
                continue
            obj_dist = np.linalg.norm(im_gt['cam_t_m2c'])
            obj_dists.append(obj_dist)

            # Camera origin in the model coordinate system
            cam_orig_m = -np.linalg.inv(im_gt['cam_R_m2c']).dot(im_gt['cam_t_m2c'])

            # Azimuth from (0, 360)
            azimuth = math.atan2(cam_orig_m[1, 0], cam_orig_m[0, 0])
            if azimuth < 0:
                azimuth += 2.0 * math.pi
            azimuths.append((180.0 / math.pi) * azimuth)

            # Elevation from (-90, 90)
            a = np.linalg.norm(cam_orig_m)
            b = np.linalg.norm([cam_orig_m[0, 0], cam_orig_m[1, 0], 0])
            elev = math.acos(b / a)
            if cam_orig_m[2, 0] < 0:
                elev = -elev
            elevs.append((180.0 / math.pi) * elev)

print('Number of images: ' + str(ims_count))

print('Min dist: ' + str(min(obj_dists)))
print('Max dist: ' + str(max(obj_dists)))

print('Min azimuth: ' + str(min(azimuths)))
print('Max azimuth: ' + str(max(azimuths)))

print('Min elev: ' + str(min(elevs)))
print('Max elev: ' + str(max(elevs)))

plt.figure('Object distance')
plt.hist(obj_dists, bins=100)

plt.figure('Azimuth')
plt.hist(azimuths, bins=100)

plt.figure('Elevation')
plt.hist(elevs, bins=100)

plt.show()
