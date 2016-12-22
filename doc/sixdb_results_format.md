# Format of results

## Problem: 6D localization of a single instance of a single object

For a test image with id=XXXX and an object with id=YY, the results of an
evaluated method are expected to be stored in file XXXX_YY.txt in the following
format:

The first line contains a single number (float) representing the run time
(in seconds, -1 if not available):

```
det_time
```

Every other line contains a 6D object pose estimate in this format:

```
object_id score r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
```

where score (float) is a confidence of the estimate, R = [r11 r12 r13;
r21 r22 r23; r31 r32 r33] is a 3x3 rotation matrix and t = [t1 t2 t3]' is
a 3x1 translation vector (in mm). P = K * [R t] is the camera matrix, which
transforms a 3D point x\_m in the model coordinate system to a 3D point x\_c
in the camera coordinate system (as defined in OpenCV): x\_c = P * x\_m.

For this evaluation problem, one estimate per file is sufficient. The one with
the highest confidence will be selected if the file contains more estimates.
There are test images with multiple instances of the same object and if more
estimates are provided, the results can be later re-used to evaluate
"6D localization of multiple instances of a single object".

Example: If a test image XXXX contains objects AA, BB and CC, the method is
expected to run three times on this image and to store results in files:
XXXX_AA.txt, XXXX_BB.txt, XXXX_CC.txt.
