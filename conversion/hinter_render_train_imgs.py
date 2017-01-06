# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import sys
import cv2
import numpy as np
import yaml

sys.path.append(os.path.abspath('..'))
from pysixdb import view_sampler, inout, misc, renderer
from params import par_hinterstoisser as par

obj_ids = range(1, 16)
# obj_ids = range(14, 16)

min_n_views = 1000 # The final number of views depends on the sampling method
# radii = range(600, 1101, 100) # [mm] 60 - 110 cm (calculated using calc_obj_dist.py)
# radii = range(400, 1501, 100) # [mm] 35 - 150 cm with extended GT
radii = [400]
# radii = [1200, 1300, 1400, 1500]
halfsphere = True # Views only from the top hemisphere
clip_near = 10 # [mm]
clip_far = 10000 # [mm]
ambient_weight = 0.8
shading = 'phong' # 'flat', 'phong'
ssaa_fact = 4 # Anti-aliasing factor (for supersampling anti-aliasing)

base_path = '/local/datasets/tlod/hinterstoisser/'
model_mpath = base_path + 'models/obj_{:02d}.ply'

# obj_info_mpath = base_path + 'train/obj_{:02d}/obj_info.yml'
# rgb_mpath = base_path + 'train/obj_{:02d}/rgb/{:04d}.png'
# depth_mpath = base_path + 'train/obj_{:02d}/depth/{:04d}.png'

obj_info_mpath = base_path + 'train_1200-1500/obj_{:02d}/obj_info.yml'
rgb_mpath = base_path + 'train_1200-1500/obj_{:02d}/rgb/{:04d}.png'
depth_mpath = base_path + 'train_1200-1500/obj_{:02d}/depth/{:04d}.png'

# Super-sampling anti-aliasing (SSAA)
# https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
# The RGB image is rendered at ssaa_fact times higher resolution and then
# down-sampled to the required resolution.
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in par.cam['im_size']]
K_rgb = par.cam['K'] * ssaa_fact

for obj_id in obj_ids:
    # Prepare folders
    misc.ensure_dir(os.path.dirname(rgb_mpath.format(obj_id, 0)))
    misc.ensure_dir(os.path.dirname(depth_mpath.format(obj_id, 0)))

    # Load model
    model_path = model_mpath.format(obj_id)
    model = inout.load_ply(model_path)

    obj_info = {}
    im_id = 0
    for radius in radii:
        # Sample views
        views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                       halfsphere)
        print('Sampled views: ' + str(len(views)))
        view_sampler.save_vis('../output/views_hinter_radius=' + str(radius) + '.ply',
                              views, views_level)

        # Render the object model from all the views
        for view_id, view in enumerate(views):
            if view_id % 10 == 0:
                print('obj,radius,view: ' + str(obj_id) +
                      ',' + str(radius) + ',' + str(view_id))

            # Render depth image
            depth = renderer.render(model, par.cam['im_size'], par.cam['K'],
                                    view['R'], view['t'],
                                    clip_near, clip_far, mode='depth')
            depth *= 10.0  # Convert depth map to [100um]

            # Render RGB image
            rgb = renderer.render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                  clip_near, clip_far, ambient_weight=ambient_weight,
                                  shading=shading, mode='rgb')
            rgb = cv2.resize(rgb, par.cam['im_size'], interpolation=cv2.INTER_AREA)

            # Save the rendered images
            inout.write_im(rgb_mpath.format(obj_id, im_id), rgb)
            inout.write_depth(depth_mpath.format(obj_id, im_id), depth)

            # Get 2D bounding box of the object model at the ground truth pose
            ys, xs = np.nonzero(depth > 0)
            obj_bb = misc.calc_2d_bbox(xs, ys, par.cam['im_size'])

            obj_info[im_id] = {
                'cam_K': par.cam['K'].flatten().tolist(),
                'cam_R_m2c': view['R'].flatten().tolist(),
                'cam_t_m2c': view['t'].flatten().tolist(),
                'obj_bb': [int(x) for x in obj_bb],
                'view_level': int(views_level[view_id]),
                # 'sphere_radius': float(radius)
            }

            im_id += 1
        #     break
        # break

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    yaml.add_representer(float, float_representer)

    # Store metadata
    with open(obj_info_mpath.format(obj_id), 'w') as f:
        yaml.dump(obj_info, f, width=10000)
