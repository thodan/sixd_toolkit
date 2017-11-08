# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys
import math
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, transform
import hinter_flip

model_mpath = '/local/datasets/tlod/hinterstoisser/models/obj_{:02d}.ply'
bbox_cens_path = 'output/hinter_bbox_cens.yml' # File to save bbox centers

bbox_cens = []
obj_ids = range(1, 16)
for obj_id in obj_ids:
    print('Processing obj: ' + str(obj_id))
    model_path = model_mpath.format(obj_id)
    model = inout.load_ply(model_path)

    # Translate the bounding box center to the origin
    bb_cen = 0.5 * (model['pts'].min(axis=0) + model['pts'].max(axis=0))
    model['pts'] -= bb_cen
    bbox_cens.append(bb_cen.flatten().tolist())

    # Rotate around Y axis by pi
    R = transform.rotation_matrix(math.pi, [0, 1, 0])[:3, :3]
    model['pts'] = R.dot(model['pts'].T).T

    # Rotate around Z axis by pi
    if hinter_flip.obj_flip_z[obj_id]:
        R = transform.rotation_matrix(math.pi, [0, 0, 1])[:3, :3]
        model['pts'] = R.dot(model['pts'].T).T

    # Save the transformed model
    inout.save_ply(model_path, model['pts'], model['colors'], model['normals'],
                   model['faces'])

with open(bbox_cens_path, 'w') as f:
    yaml.dump(bbox_cens, f, width=10000)
