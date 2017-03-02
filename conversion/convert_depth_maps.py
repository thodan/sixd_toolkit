# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath('..'))
from pysixdb import inout
# from params import par_hinterstoisser as par
# from params import par_tejani as par
from params import par_doumanoglou as par

# data_ids = range(1, par.obj_count + 1)
data_ids = range(1, par.scene_count + 1)

# depth_mpath = par.train_depth_mpath
depth_mpath = par.test_depth_mpath

scale = 0.1

for data_id in data_ids:
    print('Processing id: ' + str(data_id))
    depth_paths = sorted(glob.glob(os.path.join(
        os.path.dirname(depth_mpath.format(data_id, 0)), '*')))
    for depth_path in depth_paths:
        d = inout.read_depth(depth_path)
        d *= scale
        d = np.round(d).astype(np.uint16)
        inout.write_depth(depth_path, d)
