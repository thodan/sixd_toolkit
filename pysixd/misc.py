# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance

def ensure_dir(path):
    """
    Ensures that the specified directory exists.

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def draw_rect(vis, rect, color=(255, 255, 255)):
    vis_pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_pil)
    draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                   outline=color, fill=None)
    del draw
    return np.asarray(vis_pil)

def project_pts(pts, K, R, t):
    assert(pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T

def depth_im_to_dist_im(depth_im, K):
    """
    Converts depth image to distance image.

    :param depth_im: Input depth image, where depth_im[y, x] is the Z coordinate
    of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the Kinect-like sensors).
    :param K: Camera matrix.
    :return: Distance image dist_im, where dist_im[y, x] is the distance from
    the camera center to the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point.
    """
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im

def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

def calc_pose_2d_bbox(model, im_size, K, R_m2c, t_m2c):
    pts_im = project_pts(model['pts'], K, R_m2c, t_m2c)
    pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox(pts_im[:, 0], pts_im[:, 1], im_size)

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def calc_pts_diameter(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    diameter = -1
    for pt_id in range(pts.shape[0]):
        if pt_id % 1000 == 0: print(pt_id)
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter
