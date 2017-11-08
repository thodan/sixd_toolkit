# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os

fpath = os.path.abspath(__file__)

if '/home/tom' in fpath:
    name = 'lopata'
elif '/home.dokt/hodanto2' in fpath:
    name = 'cmpgrid'
elif '/home/hodan' in fpath:
    name = 'tutgrid'
else:
    name = None
    print('Error: Unknown machine!')
    exit(-1)

print('Running on ' + name)
