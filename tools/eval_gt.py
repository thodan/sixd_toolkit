import os
import sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from pysixdb import inout, misc, renderer

from params import par_hinterstoisser as par
# from params import par_tejani as par

# scene_ids = range(1, 7)
scene_ids = [15]
vis_depth = False
vis_rgb = True

out_dir = os.path.join('..', 'output', 'eval_gt', par.dataset_name)
misc.ensure_dir(out_dir)

stats = []
for scene_id in scene_ids:
    scene_id_str = str(scene_id).zfill(2)

    # Load info about the test images (including camera parameters etc.)
    scene_info = inout.load_scene_info(par.scene_info_mpath.format(scene_id))

    # Load the ground truth poses
    gts = inout.load_scene_gt(par.scene_gt_mpath.format(scene_id))

    # Load models of objects present in the scene
    scene_obj_ids = set()
    for gt in gts[0]:
        scene_obj_ids.add(gt['obj_id'])
    models = {}
    for scene_obj_id in scene_obj_ids:
        model_path = par.model_mpath.format(scene_obj_id)
        models[scene_obj_id] = inout.load_ply(model_path)

    # Go through the scene images
    for im_id, im_info in scene_info.items():
        if im_id % 50 != 0: continue
        if im_id % 50 == 0:
            print('scene, image: ', scene_id, im_id)
        im_id_str = str(im_id).zfill(4)

        # Load depth image
        test_depth_path = par.test_depth_mpath.format(scene_id, im_id)
        test_depth = scipy.misc.imread(test_depth_path).astype(np.float)
        test_depth *= 0.1 # to [mm]

        # Load RGB image
        if vis_rgb:
            test_rgb_path = par.test_rgb_mpath.format(scene_id, im_id)
            test_rgb = scipy.misc.imread(test_rgb_path).astype(np.float)

        im_size = (test_depth.shape[1], test_depth.shape[0])
        K = im_info['cam_K']

        for gt_id, gt in enumerate(gts[im_id]):
            gt_id_str = str(gt_id).zfill(2)
            model = models[gt['obj_id']]
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            color = (1.0, 0.0, 0.0)

            ren_depth = renderer.render(model, im_size, K, R, t, mode='depth')
            ren_depth = ren_depth.astype(np.float)

            mask = (test_depth > 0) * (ren_depth > 0)
            diff_im = mask.astype(np.float) * (test_depth - ren_depth)
            stats.append({
                'scene_id': scene_id,
                'im_id': im_id,
                'gt_id': gt_id,
                'diff': diff_im[mask]
            })

            if vis_depth:
                plt.matshow(diff_im)
                plt.colorbar()
                plt.title('sensed - gt [mm]')
                vis_path = os.path.join(out_dir, scene_id_str + '_' + im_id_str +
                                        '_' + gt_id_str + '_depth_diff.jpg')
                plt.savefig(vis_path, pad=0)
                plt.close()

            if vis_rgb:
                ren_rgb = renderer.render(model, im_size, K, R, t,
                                          surf_color=color, mode='rgb')
                ren_rgb = misc.draw_rect(ren_rgb, gt['obj_bb'])

                vis_rgb_im = 0.5 * ren_rgb + 0.5 * test_rgb
                vis_rgb_im[vis_rgb_im > 255] = 255
                vis_rgb_im = vis_rgb_im.astype(np.uint8)

                vis_path = os.path.join(out_dir, scene_id_str + '_' + im_id_str +
                                        '_' + gt_id_str + '.jpg')
                scipy.misc.imsave(vis_path, vis_rgb_im)

# Analyse the depth differences
trim_fract = 0.3 # Fraction of depth differences to be trimmed
diff_all = []
diff_means = []
diff_stds = []
for stat in stats:
    min_keep_count = int(len(stat['diff']) * (1.0 - trim_fract))
    diff_trim = stat['diff'][np.argsort(np.abs(stat['diff']))[:min_keep_count]]

    diff_all += list(diff_trim)
    diff_means.append(np.mean(diff_trim))
    diff_stds.append(np.std(diff_trim))

print('diff_all mean: ' + str(np.mean(diff_all)))
print('diff_all std: ' + str(np.std(diff_all)))

inds_order = np.argsort(np.abs(diff_means))[::-1]
# inds_order = np.argsort(np.abs(diff_stds))[::-1]

# import pickle
# good_ids_path = '/local/datasets/tlod/imperial/tejani_filtering/tejani/good_ids.p'
# with open(good_ids_path, 'r') as f:
#     good_ids = pickle.load(f)
# good_ids_scenes = {}
# for good_id in good_ids:
#     if good_id[0] not in good_ids_scenes.keys():
#         good_ids_scenes[good_id[0]] = [good_id[1]]
#     else:
#         good_ids_scenes[good_id[0]].append(good_id[1])

# for i in range(300):
for i in range(len(inds_order)):
    ind = inds_order[i]
    stat = stats[ind]
    #if stat['im_id'] in good_ids_scenes[stat['scene_id']]:
    print('scene_id: {}, im_id: {}, gt_id: {}, mean diff: {}, std diff: {}'.format(
        stat['scene_id'], stat['im_id'], stat['gt_id'], diff_means[ind], diff_stds[ind]))

plt.hist(diff_means, bins=20)
plt.show()
