# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
from PIL import Image, ImageDraw
import numpy as np

def ensure_dir(path):
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
