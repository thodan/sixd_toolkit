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

## Installation

### Dependencies

To install the required python packages, run:

```
pip install -r requirements.txt
```

In the case of problems, try to run ```pip install --upgrade pip setuptools```
first.

### Rendering

Rendering is implemented using the Glumpy library and was tested with the GLFW
library as the window backend. In Linux, you can install the GLFW library with:

```
apt-get install libglfw3
```

To use a different backend library, see the first lines of
**pysixd/renderer.py**.

## Evaluation

1. Run your method on the SIXD datasets and prepare the results in
[this format](https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_results_format.md).
2. Run **tools/eval_calc_errors.py** to calculate errors of the pose estimates
(fill list **result_paths** in the script with paths to the results first).
3. Run **tools/eval_loc.py** to calculate performance scores in the
6D localization task (fill list **error_paths** in the script with paths to the
calculated errors first).

Sample results can be downloaded from:
http://ptak.felk.cvut.cz/6DB/public/sixd_results

**Note: The setting of evaluation parameters for the SIXD Challenge 2017 is
currently being decided.**

## Author

**Tomas Hodan**
- hodantom@cmp.felk.cvut.cz
- http://www.hodan.xyz
- Center for Machine Perception, Czech Technical University in Prague
