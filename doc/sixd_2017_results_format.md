# SIXD challenge 2017: format of results


For a test image XXXX and object YY, which is present in that image, the
evaluated method is expected to estimate the 6D pose of an instance of object YY
and save the results in file **XXXX\_YY.txt**.

The first line of the file contains the run time represented as a single real
number (in seconds, -1 if not available). Every other line contains a 6D object
pose estimate in the following format:

```
object_id score r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
```

where score is a confidence of the estimate (the range of score is not
restricted), R = [r11 r12 r13; r21 r22 r23; r31 r32 r33] is a 3x3 rotation
matrix saved row-wise, and t = [t1 t2 t3]' is a 3x1 translation vector (in mm).
P = K * [R t] is the camera matrix that transforms 3D point p\_m = [x, y, z, 1]'
in the model coordinate system to 2D point p\_i = [u, v, 1]' in the image
coordinate system: s * p\_i = P * p\_m. The camera coordinate system is as
defined in OpenCV with the camera looking along the Z axis. K is provided with
the test images.

We encourage the participants to provide more estimates per file. This will
allow us to evaluate 1) top-N recall (i.e. Is there a correct estimate among the
N with the highest score?), and 2) 6D localization of multiple instances of a
single object, which is one of the tasks considered for future editions of the
challenge.

All test images are used for the evaluation, even those with multiple instances
of the object of interest. The list of objects that are present in an image can
be obtained from file gt.yml.

The files with results are expected in this structure:

**METHOD\_DATASET[\_TESTTYPE]/ZZ/XXXX\_YY.txt**

METHOD is the name of your method, DATASET is the dataset name (hinterstoisser,
t-less, tud_light, rutgers, tejani, or doumanoglou), TESTTYPE is the data type
(used only for datasets with more data types), and ZZ, XXXX, and YY is the ID of
the test scene, the test image and the object.


### Example

Test image 0000 (let us consider the image from the Primesense sensor) of test
scene 01 from the T-LESS dataset contains objects 02, 25, 29 and 30. A method
is expected to run three times on this image and store the results in files:

- mymethod_t-less_primesense/01/0000\_02.txt
- mymethod_t-less_primesense/01/0000\_25.txt
- mymethod_t-less_primesense/01/0000\_29.txt
- mymethod_t-less_primesense/01/0000\_30.txt

Example content of file 0000_25.txt:
```
1.6553
25 2.591 0.164305 -0.116608 -0.979493 -0.881643 0.427981 -0.198842 0.442391 0.896233 -0.0324873 -45.7994 75.008 801.078
25 3.495 -0.321302 0.937843 -0.131205 -0.926472 -0.282636 0.248529 0.195999 0.201411 0.959697 -77.2591 -23.8807 770.016
25 0.901 0.133352 -0.546655 -0.826671 0.244205 0.826527 -0.507166 0.960511 -0.134245 0.243715 79.4697 -23.619 775.376
25 1.339 -0.998023 0.0114256 0.061813 -1.55661e-05 -0.983388 0.181512 0.0628601 0.181152 0.981445 8.9896 75.8737 751.272
25 1.512 0.211676 0.12117 -0.969799 0.120886 0.981419 0.149007 0.969835 -0.148776 0.193096 7.10206 -53.5385 784.077
25 0.864 0.0414156 -0.024525 -0.998841 0.295721 0.955208 -0.0111921 0.954376 -0.294915 0.046813 40.1253 -34.8206 775.819
25 1.811 0.0369952 -0.0230957 -0.999049 0.304581 0.952426 -0.0107392 0.951768 -0.303894 0.0422696 36.5109 -27.5895 775.758
25 1.882 0.263059 -0.946784 0.18547 -0.00346377 -0.193166 -0.98116 0.964774 0.25746 -0.0540936 75.3467 -28.4081 771.788
25 1.195 -0.171041 -0.0236642 -0.98498 -0.892308 0.427616 0.144675 0.41777 0.90365 -0.0942557 -69.8224 73.1472 800.083
25 1.874 0.180726 -0.73069 0.658354 0.0538221 -0.661026 -0.74843 0.98206 0.170694 -0.0801374 19.7014 -68.7299 781.424
```


### Documentation

If you want your results to be included in a publication about the challenge,
a documentation of the evaluation is required. It is expected in file
**METHOD\_DATASET[\_TESTTYPE]/eval_doc.txt** with this format:

```
SIXD challenge 2017 - documentation of the evaluation
Author: ...
Method: ... (any related publications)
PC tech specs: ... (for comparison of the running times)
Dataset: ...
Training input:
    Object model type: ... (default, cad, reconst, none)
    Image type: ... (real, rendered, which light conditions for TUD Light, etc.)
    Number of images per object: ...
Test input:
    Modality: ... (RGB-D, RGB or D)
Other info: ...
...
```
