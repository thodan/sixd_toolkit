# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
import struct


def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: A path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split(' ')[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split(' ')[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split(' ')[-1], line.split(' ')[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            face_props.append(('n_corners', elems[2]))
            for i in range(face_n_corners):
                face_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print 'Error: Only triangular faces are supported.'
                        print 'Number of face corners:', val
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print 'Error: Only triangular faces are supported.'
                        print 'Number of face corners:', int(elems[prop_id])
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model

def save_ply(path, pts, pts_colors=np.array([]), pts_normals=np.array([]), faces=np.array([])):
    """
    Saves a 3D mesh model to a PLY file.

    :param path: A path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors: nx3 ndarray
    :param pts_normals: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert(len(pts) == len(pts_colors))

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, 'w')
    f.write(
        'ply\n'
        'format ascii 1.0\n'
        #'format binary_little_endian 1.0\n'
        'element vertex ' + str(valid_pts_count) + '\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    if pts_normals.size != 0:
        f.write(
            'property float nx\n'
            'property float ny\n'
            'property float nz\n'
        )
    if pts_colors.size != 0:
        f.write(
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
        )
    if faces.size != 0:
        f.write(
            'element face ' + str(len(faces)) + '\n'
            'property list uchar int vertex_indices\n'
        )
    f.write('end_header\n')

    format_float = "{:.4f}"
    format_3float = " ".join((format_float for _ in range(3)))
    format_int = "{:d}"
    format_3int = " ".join((format_int for _ in range(3)))
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(format_3float.format(*pts[pt_id].astype(float)))
            if pts_normals.size != 0:
                f.write(' ')
                f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
            if pts_colors.size != 0:
                f.write(' ')
                f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
            f.write('\n')
    for face in faces:
        f.write(' '.join(map(str, map(int, [len(face)] + list(face.squeeze())))) + ' ')
        f.write('\n')
    f.close()
