# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Example of views generation

import numpy as np
from pysixdb import view_sampler, inout

min_n_views = 500
radius = 1
halfsphere = True

# Sample views
views = view_sampler.sample_views(min_n_views, radius, halfsphere)
print('Sampled views: ' + str(len(views)))

# Visualization (saved as a PLY file)
pts = []
normals = []
colors = []
for view_id, view in enumerate(views):
    R_inv = np.linalg.inv(view['R'])
    pts += [R_inv.dot(-view['t']).squeeze(),
            R_inv.dot(np.array([[0.01, 0, 0]]).T - view['t']).squeeze(),
            R_inv.dot(np.array([[0, 0.01, 0]]).T - view['t']).squeeze(),
            R_inv.dot(np.array([[0, 0, 0.01]]).T - view['t']).squeeze()]

    normal = R_inv.dot(np.array([0, 0, 1]).reshape((3, 1)))
    normals += [normal.squeeze(),
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
                np.array([0, 0, 0])]

    intens = 255 * view_id / float(len(views))
    colors += [[intens, intens, intens],
               [255, 0, 0],
               [0, 255, 0],
               [0, 0, 255]]

inout.save_ply('output/sphere.ply',
               pts=np.array(pts),
               pts_normals=np.array(normals),
               pts_colors=np.array(colors))
