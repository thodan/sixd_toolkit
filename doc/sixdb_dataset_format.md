# 6DB standard dataset format

## Directory structure

The dataset is organized as follows:

* **models[\_MODELTYPE]/obj\_YY.ply** - 3D model of object YY.
* **train[\_SENSORTYPE]/obj\_YY/{rgb,depth}/XXXX.png** - Training images of object YY.
* **test[\_SENSORTYPE]/scene\_ZZ/{rgb,depth}/XXXX.png** - Test images of scene ZZ.

MODELTYPE and SENSORTYPE is optional and is used if more types are available.
XXXX is ID (a zero-padded number) of an image.


## Training images

The training images of each object are accompanied with file obj\_info.yml
(located in train[\_SENSORTYPE]/obj\_YY), that contains for each image the
following information:

* **cam\_K** - 3x3 intrinsic camera matrix K (saved row-wise)
* **cam\_R\_m2c** - 3x3 rotation matrix R\_m2c (saved row-wise)
* **cam\_t\_m2c** - 3x1 translation vector t\_m2c

P\_m2c = K * [R\_m2c, t\_m2c] is the camera matrix which transforms a 3D point x\_m
in the model coordinate system to a 3D point x\_c in the camera coordinate
system: x\_c = P * x\_m.

Note: The matrix K can be different for each image if the provided images
were obtained by cropping of the captured images (i.e. the principal point is
not constant).


## Test images

The test images from each scene are accompanied with file scene\_info.yml
(located in test[\_SENSORTYPE]/scene\_ZZ), that contains for each image
the following information:

* **cam\_K** - 3x3 intrinsic camera matrix K (saved row-wise)

Note: The matrix K can be different for each image for the same reason as in the
case of training images.

The ground truth poses of the objects which are present in the scene are
provided in file scene\_gt.yml (located in test[\_SENSORTYPE]/scene\_ZZ),
that contains for each image and object the following information:

* **obj\_id** - Object ID
* **cam\_R\_m2c** - 3x3 rotation matrix R\_m2c (saved row-wise)
* **cam\_t\_m2c** - 3x1 translation vector t\_m2c

P\_m2c = K * [R\_m2c, t\_m2c] is the camera matrix which transforms a 3D point x\_m
in the model coordinate system to a 3D point x\_c in the camera coordinate
system: x\_c = P * x\_m.


## 3D object models

The 3D object models are provided in PLY (ascii) format, with vertex normals,
and ideally also with vertex color.

Note: The vertex normals can be calculated using the MeshLab implementation
(http://meshlab.sourceforge.net/) of the following method:
G. Thurrner and C. A. Wuthrich, Computing vertex normals from polygonal facets,
Journal of Graphics Tools 3.1 (1998).


## Coordinate systems

The center of the bounding box of the object model is aligned to the origin
of the model coordinate system. The Z coordinate is pointing up (when the
object is seen standing "naturally up-right").

The camera coordinate system is as in OpenCV:
http://docs.opencv.org/2.4/modules/calib3d/doc/camera\_calibration\_and\_3d\_reconstruction.html


## Units

* Depth images: 0.1 mm
* 3D object models: 1 mm
* Translation vectors: 1 mm
