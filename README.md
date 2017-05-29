# SIXD Toolkit

Python scripts to facilitate participation in the SIXD Challenge:
http://cmp.felk.cvut.cz/sixd/challenge_2017

- **conversion** - Scripts used to convert the datasets from the original format
                   to the SIXD standard format.
- **doc** - Documentation and conventions.
- **params** - Parameters (paths to datasets etc.) used by other scripts.
- **pysixd** - Core library that takes care of i/o operations, rendering,
               calculation of pose errors etc.
- **tools** - Scripts for evaluation, rendering of training images,
              visualization of 6D object poses etc.

## Dependencies

To install the required python packages, run:

```
pip install -r requirements.txt
```

In the case of problems, try to run ```pip install --upgrade pip setuptools```
first.

## Rendering

Rendering is implemented using the Glumpy library and was tested with the GLFW
library as the window backend. In Linux, you can install the GLFW library with:

```
apt-get install libglfw3
```

To use a different backend library, see the first lines of pysixd/renderer.py.

## Author

**Tomas Hodan**
- hodantom@cmp.felk.cvut.cz
- http://www.hodan.xyz
- Center for Machine Perception, Czech Technical University in Prague
