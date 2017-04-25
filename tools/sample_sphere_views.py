# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Example of sampling views from a view sphere.

import os
import sys

sys.path.append(os.path.abspath('..'))
from pysixd import view_sampler, misc

min_n_views = 642
radius = 1
hemisphere = False

out_views_vis_path = '../output/view_sphere.ply'

misc.ensure_dir(os.path.dirname(out_views_vis_path))

# Sample views
views, views_level = view_sampler.sample_views(min_n_views, radius, hemisphere)
print('Sampled views: ' + str(len(views)))

view_sampler.save_vis(out_views_vis_path, views)
