import numpy as np

from utils import *

DATA_FILE = 'Fourier/Data.dat'
FLOW_FILE = 'Fourier/Flowfield.res'


def write_data(height, depth, period_or_length, current=0, N=N, length_str='Period', current_crit=1, height_steps=1):
    '''
    '''
    s = f'JP Test Wave\n' \
    f'{height / depth}  H/d\n' \
    f'{length_str}  Measure of length: "Wavelength" or "Period"\n' \
    f'{period_or_length / depth if length_str == "Wavelength" else period_or_length * np.sqrt(G / depth)}  Value of that length: L/d or T(g/d)^1/2 respectively\n' \
    f'{current_crit}  Current criterion (1 or 2)\n' \
    f'{current / np.sqrt(G * depth)}  Current magnitude, (dimensionless) ubar/(gd)^1/2\n' \
    f'{N} Number of Fourier components or Order of Stokes/cnoidal theory\n' \
    f'{height_steps} Number of height steps to reach H/d\n' \
    'FINISH\n'
    
    with open(DATA_FILE, 'w') as f: f.write(s)


def read_flow_data(depth, ref_depth):
    '''
    '''
    with open(FLOW_FILE, 'r') as f:
        lines = f.readlines()
    
    res = []
    reading = False
    for l in lines:
        l = l.strip()
        if l and not l.startswith('#'):
            reading = True
            res.append([float(x) for x in l.split()])
        if reading and l.startswith('#'):
            break

    res = np.array(res)[:, :2]
    res[:, 0] *= depth
    res[:, 1] *= np.sqrt(G * depth)

    return np.interp(ref_depth, res[:, 0], res[:, 1])

