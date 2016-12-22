# Format of results

## Problem: 6D localization of a single instance of a single object

For each test image with id=XXXX, the evaluated method is supposed to store
its results into file "XXXX.est" with the following format:

The first line contains a single float number representing the detection time
(-1 if not available):

det\_time

Each of the other lines has this format:

object\_id score r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3

where R = [r11 r12 r13; r21 r22 r23; r31 r32 r33] is a 3x3 rotation matrix and
t = [t1 t2 t3]' is a 3x1 translation vector. P = K * [R t] is the camera matrix,
which transforms a 3D point x\_m in the model coordinate system to a 3D point
x\_c in the camera coordinate system: x\_c = P * x\_m.
