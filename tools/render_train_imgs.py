# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Renders RGB-D images of an object model from a "uniformly" sampled view sphere.

import os
import sys
import math
import yaml
import numpy as np
import cv2

sys.path.append(os.path.abspath('..'))
from pysixd import view_sampler, inout, misc, renderer
from params import par_rutgers as par

# dataset = 'hinterstoisser'
# dataset = 'tejani'
# dataset = 'rutgers'
dataset = 'tud_light'


if dataset == 'hinterstoisser':
    from params import par_hinterstoisser as par

    # Range of object dist. in test images: 346.31 - 1499.84 mm - with extended GT
    # (there are only 3 occurrences under 400 mm)
    # Range of object dist. in test images: 600.90 - 1102.35 mm - with only original GT
    radii = [400] # Radii of the view sphere [mm]
    # radii = range(600, 1101, 100)
    # radii = range(400, 1501, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

elif dataset == 'tejani':
    from params import par_tejani as par

    # Range of object dist. in test images: 509.12 - 1120.41 mm
    radii = [500] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100)

    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

elif dataset == 'rutgers':
    from params import par_rutgers as par

    # Range of object distances in test images: 594.41 - 739.12 mm
    radii = [590] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100) # [mm]

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-0.5 * math.pi, 0.5 * math.pi)

elif dataset == 'tud_light':
    from params import par_tud_light as par

    # Range of object distances in test images: 851.29 - 2016.14 mm
    radii = [850] # Radii of the view sphere [mm]
    # radii = range(500, 1101, 100) # [mm]

    azimuth_range = (0, 2 * math.pi)
    elev_range = (-0.4363, 0.5 * math.pi) # (-25, 90) [deg]


# Objects to render
obj_ids = range(1, par.obj_count + 1)

# Minimum required number of views on the whole view sphere. The final number of
# views depends on the sampling method.
min_n_views = 1000

clip_near = 10 # [mm]
clip_far = 10000 # [mm]
ambient_weight = 0.8 # Weight of ambient light [0, 1]
shading = 'phong' # 'flat', 'phong'

# Super-sampling anti-aliasing (SSAA)
# https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
# The RGB image is rendered at ssaa_fact times higher resolution and then
# down-sampled to the required resolution.
ssaa_fact = 4

# Output path masks
out_rgb_mpath = '../output/render/{:02d}/rgb/{:04d}.png'
out_depth_mpath = '../output/render/{:02d}/depth/{:04d}.png'
out_obj_info_path = '../output/render/{:02d}/info.yml'
out_obj_gt_path = '../output/render/{:02d}/gt.yml'
out_views_vis_mpath = '../output/render/views_radius={}.ply'

# Prepare output folder
# misc.ensure_dir(os.path.dirname(out_obj_info_path))

# Image size and K for SSAA
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in par.cam['im_size']]
K_rgb = par.cam['K'] * ssaa_fact

for obj_id in obj_ids:
    # Prepare folders
    misc.ensure_dir(os.path.dirname(out_rgb_mpath.format(obj_id, 0)))
    misc.ensure_dir(os.path.dirname(out_depth_mpath.format(obj_id, 0)))

    # Load model
    model_path = par.model_mpath.format(obj_id)
    model = inout.load_ply(model_path)

    # Load model texture
    if par.model_texture_mpath:
        model_texture_path = par.model_texture_mpath.format(obj_id)
        model_texture = cv2.imread(model_texture_path)
        model_texture = cv2.cvtColor(model_texture, cv2.COLOR_RGB2BGR)
    else:
        model_texture = None

    obj_info = {}
    obj_gt = {}
    im_id = 0
    for radius in radii:
        # Sample views
        views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                       azimuth_range, elev_range)
        print('Sampled views: ' + str(len(views)))
        view_sampler.save_vis(out_views_vis_mpath.format(str(radius)),
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

            # Convert depth is in the same units as for test images
            depth /= par.cam['depth_scale']

            # Render RGB image
            rgb = renderer.render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                  clip_near, clip_far, texture=model_texture,
                                  ambient_weight=ambient_weight, shading=shading,
                                  mode='rgb')
            rgb = cv2.resize(rgb, par.cam['im_size'], interpolation=cv2.INTER_AREA)

            # Save the rendered images
            inout.write_im(out_rgb_mpath.format(obj_id, im_id), rgb)
            inout.write_depth(out_depth_mpath.format(obj_id, im_id), depth)

            # Get 2D bounding box of the object model at the ground truth pose
            ys, xs = np.nonzero(depth > 0)
            obj_bb = misc.calc_2d_bbox(xs, ys, par.cam['im_size'])

            obj_info[im_id] = {
                'cam_K': par.cam['K'].flatten().tolist(),
                'view_level': int(views_level[view_id]),
                #'sphere_radius': float(radius)
            }

            obj_gt[im_id] = [{
                'cam_R_m2c': view['R'].flatten().tolist(),
                'cam_t_m2c': view['t'].flatten().tolist(),
                'obj_bb': [int(x) for x in obj_bb],
                'obj_id': int(obj_id)
            }]

            im_id += 1

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    yaml.add_representer(float, float_representer)

    # Store metadata
    with open(out_obj_info_path.format(obj_id), 'w') as f:
        yaml.dump(obj_info, f, width=10000)

    with open(out_obj_gt_path.format(obj_id), 'w') as f:
        yaml.dump(obj_gt, f, width=10000)
