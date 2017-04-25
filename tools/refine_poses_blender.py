"""
Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
Center for Machine Perception, Czech Technical University in Prague

A script to facilitate refinement of ground truth 6D object poses in Blender.

1) Open Blender console and run:
----------
import imp
script_path = '/home/tom/th_data/cmp/projects/sixd/sixd_toolkit/tools/refine_poses_blender.py'
p = imp.load_source("b", script_path)
p.init(bpy)
----------

2) To load scene with ID=scene_id, run:
----------
p.load_scene(scene_id)
----------
It will load camera parameters from "scene_info.yml" and ground truth poses
from "scene_gt.yml" - paths to these files are specified in the parameter
setting files in folder "params". The command loads the object models at poses
expressed in the world coordinate system - it takes poses defined for image with
ID=ref_im_ind and transforms them to the world coordinate system using the known
camera-to-world transformation loaded for that image from "scene_info.yml".

3) To save the refined object poses (they will be saved back to "scene_gt.yml"
and will overwrite the file content), run:
----------
p.save_scene(scene_id)
----------
The poses are transformed from the world coordinate system to the camera
coordinate systems using the camera-to-world transformations that are assumed
known for all scene images (loaded from the "scene_info.yml" file). For each
image, it also saves a 2D bounding box for each of the objects.

Useful for work in Blender:
Ctrl+Alt+Q - toggle quad mode
Home - view all objects
"""

import os
import sys
import numpy as np

sixd_toolkit_path = '/home/tom/th_data/cmp/projects/sixd/sixd_toolkit'
sys.path.append(os.path.abspath(sixd_toolkit_path))
from pysixd import inout, misc, transform

# Dataset parameters
# from params import par_hinterstoisser as par
# from params import par_tejani as par
from params import par_doumanoglou as par

# Path to a 3D model of the scene
#scene_model_mpath = None

ref_im_ind = 0
scene_gt = {}
scene_info = {}

bpy = None # pointer to blender's data structure
def init(bpy_in):
    """
    Initializes Blender project.
    """
    print('Init...')
    global bpy
    bpy = bpy_in
    clean()

    # Switch the 3D viewport to Textured mode
    for area in bpy.context.screen.areas:  # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            area.spaces.active.show_textured_solid = True
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    #space.viewport_shade = 'TEXTURED'
                    space.viewport_shade = 'SOLID'

def clean():
    """
    Cleans Blender project.
    """
    # Select all objects
    for obj in bpy.data.objects:
        obj.select = True

    # Remove all selected objects
    bpy.ops.object.delete()

    # Meshes are still in memory, remove them
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

def add_obj(obj_id):
    """
    Adds the specified object to the scene.
    """
    model_path = par.model_mpath.format(obj_id)
    print('Loading model {}: {}'.format(obj_id, model_path))
    bpy.ops.import_mesh.ply(filepath=model_path)

def is_obj(name):
    """
    Checks if the specified element is an object.
    """
    non_obj_names = ['World', 'Camera', 'Lamp', 'RenderLayers', 'scene']
    for non_obj_name in non_obj_names:
        if non_obj_name in name:
            return False
    return True

def get_pose(obj_name):
    """
    Returns pose of the specified object.
    """
    obj = bpy.data.objects[obj_name]
    if obj.rotation_mode.lower() != 'xyz':
        print('Error: Rotation mode of {} is not XYZ.'.format(obj_name))
        return
    return list(obj.rotation_euler), list(obj.location)

def load_scene(scene_id):
    """
    Loads the specified scene.
    """
    clean()

    global ref_im_ind
    global scene_info
    global scene_gt

    # Load scene info
    scene_info_path = par.scene_info_mpath.format(scene_id)
    print('Loading scene info: ' + scene_info_path)
    scene_info = inout.load_info(scene_info_path)
    ref_im_id = sorted(scene_info.keys())[ref_im_ind]
    scene_info_ref = scene_info[ref_im_id]
    R_w2c = scene_info_ref['cam_R_w2c']
    R_w2c_inv = np.linalg.inv(R_w2c)
    t_w2c = scene_info_ref['cam_t_w2c']

    # Load ground truth poses for the reference camera coordinate system
    scene_gt_path = par.scene_gt_mpath.format(scene_id)
    print('Loading GT poses: ' + scene_gt_path)
    scene_gt = inout.load_gt(scene_gt_path)
    scene_gt_ref = scene_gt[ref_im_id]

    # Load scene model
    #bpy.ops.import_mesh.ply(filepath=scene_model_mpath.format(scene_id))

    # Load models of objects that are present in the scene
    for gt in scene_gt_ref:
        model_path = par.model_mpath.format(gt['obj_id'])
        print('Loading model {}: {}'.format(gt['obj_id'], model_path))
        bpy.ops.import_mesh.ply(filepath=model_path)

    # Take poses from the reference camera coordinate system and transform them
    # to the world coordinate system (using the known camera-to-world trans.)
    objs = list(bpy.data.objects)
    objs_moved = [False for _ in objs]
    for gt in scene_gt_ref:
        for i, obj in enumerate(objs):
            obj_id = int(obj.name.split('obj_')[1].split('.')[0])

            if gt['obj_id'] == obj_id and not objs_moved[i]:
                print('Setting pose of model {}'.format(gt['obj_id']))
                #print(i, gt, obj_id, objs_moved[i], obj.location)

                R_m2w = R_w2c_inv.dot(gt['cam_R_m2c'])
                t_m2w = R_w2c_inv.dot(gt['cam_t_m2c']) - R_w2c_inv.dot(t_w2c)

                obj.rotation_euler = transform.euler_from_matrix(R_m2w, 'sxyz')
                obj.location = t_m2w.flatten().tolist()
                objs_moved[i] = True
                break

    # Align view to the loaded elements
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = bpy.context.copy()
            ctx['area'] = area
            ctx['region'] = area.regions[-1]
            bpy.ops.view3d.view_selected(ctx) # points view
            # bpy.ops.view3d.camera_to_view_selected(ctx) # points camera

    print("Scene loaded.")

def save_scene(scene_id):
    """
    Saves the specified scene.
    """
    global scene_info
    global scene_gt

    # Collect poses expressed in the world coordinate system
    ref_obj_poses = []
    for obj in bpy.data.objects:
        if is_obj(obj.name):
            obj_id = obj.name.split('obj_')[1].split('.')[0] # Get object ID
            e, t = get_pose(obj.name)
            R = transform.euler_matrix(e[0], e[1], e[2], axes='sxyz')[:3, :3]
            ref_obj_poses.append({
                'obj_id': int(obj_id),
                'cam_R_m2w': R,
                'cam_t_m2w': np.array(t).reshape((3, 1))
            })

    # Load models of objects present in the scene
    obj_ids = set([p['obj_id'] for p in ref_obj_poses])
    models = {}
    for obj_id in obj_ids:
        models[obj_id] = inout.load_ply(par.model_mpath.format(obj_id))

    # Transform the poses to the camera coordinate systems using the known
    # camera-to-world transformations
    for im_id in scene_gt.keys():
        scene_gt[im_id] = []
        K = scene_info[im_id]['cam_K']
        R_w2c = scene_info[im_id]['cam_R_w2c']
        t_w2c = scene_info[im_id]['cam_t_w2c']
        for pose in ref_obj_poses:
            R_m2c_new = R_w2c.dot(pose['cam_R_m2w'])
            t_m2c_new = R_w2c.dot(pose['cam_t_m2w']) + t_w2c

            # Get 2D bounding box of the projection of the object model at
            # the refined ground truth pose
            pts_im = misc.project_pts(models[int(obj_id)]['pts'], K,
                                      R_m2c_new, t_m2c_new)
            pts_im = np.round(pts_im).astype(np.int)
            ys, xs = pts_im[:, 1], pts_im[:, 0]
            obj_bb = misc.calc_2d_bbox(xs, ys, par.test_im_size)

            scene_gt[im_id].append({
                'obj_id': int(obj_id),
                'obj_bb': obj_bb,
                'cam_R_m2c': R_m2c_new,
                'cam_t_m2c': t_m2c_new
            })

    # Save the updated ground truth poses
    scene_gt_path = par.scene_gt_mpath.format(scene_id)
    print('Saving GT poses: ' + scene_gt_path)
    inout.save_gt(scene_gt_path, scene_gt)
