# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Samples poses from a view sphere.

import math
import random
import numpy as np

# Parameters
elev_steps = 1
azimuth_steps = 1
cell_samples = 1
inplane_steps = 1
inplane_max_angle = 1
z_shift = 1

def slice_sample_angle(x_min, x_max, x_last):
    if x_last > x_max or x_last < x_min:
        x_last = random.uniform(x_min, x_max)
    y = math.cos(x_last)
    y_sample = random.uniform(0, y)
    x_limit = min(math.acos(y_sample), x_max)
    x_sample = random.uniform(x_min, x_limit)
    return x_sample

elev_angle = -1.0
for elev in range(elev_steps):
    for azimuth in range(azimuth_steps):
        for inplane in range(inplane_steps):
            for cell_id in range(cell_samples):
                start = math.asin(elev / float(elev_steps))
                end = math.asin((elev + 1) / float(elev_steps))

                # Sample elevation
                elev_angle = slice_sample_angle(start, end, elev_angle)

                # Sample azimuth
                extent = (2 * math.pi) / float(azimuth_steps)
                azimuth_angle = azimuth * extent + random.uniform(0, extent)

                # Sample in-plane rotation
                extent = inplane_max_angle * 2.0 / float(inplane_steps)
                inplane_angle_start = -inplane_max_angle + extent * inplane
                inplane_angle_end = inplane_angle_start + extent
                inplane_angle = random.uniform(inplane_angle_start, inplane_angle_end)

                rvI = [0, 0, inplane_angle, 0, 0, 0]
                #6DPose hIn(rvI);

                rv_ = [elev_angle, 0, 0, 0, 0, 0]
                #6DPose hAng(rv_);

                rv = [0, azimuth_angle, 0, 0, 0, 0]
                #6DPose hAngB(rv);

                # Combine rotations around axis into one rotation and set translation
                #6DPose pose(hIn.getTransformation() * hAng.getTransformation() * hAngB.getTransformation())
                #pose.setTranslation(cv::Point3d(0.0, 0.0, z_shift * 1000.0))
