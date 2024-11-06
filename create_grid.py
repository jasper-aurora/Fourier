import numpy as np
import subprocess as sp
import itertools
from scipy import interpolate
from scipy import optimize
import os
import raschii
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


DATA_FILE = 'Fourier/Data.dat'
FLOW_FILE = 'Fourier/Flowfield.res'


G = 9.81
N = 22

STEP = 20


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


def str_after(s, sub_str):
    '''
    '''
    i = s.index(sub_str) + len(sub_str)
    return s[i:].split()[0]


def enum_product(*args):
    '''
    '''
    return itertools.product(*[enumerate(x) for x in args])


def get_interpolator(path='data', filled=True):
    '''
    '''
    height = np.load(os.path.join(path, 'height.npy'))
    depth = np.load(os.path.join(path, 'depth.npy'))
    period = np.load(os.path.join(path, 'period.npy'))
    current = np.load(os.path.join(path, 'current.npy'))
    ref_depth = np.load(os.path.join(path, 'ref_depth.npy'))
    flows = np.load(os.path.join(path, 'flows_filled2.npy' if filled else 'flows.npy'))

    # flows[flows == 0] = np.nan

    return interpolate.RegularGridInterpolator((height, depth, period, current, ref_depth), flows)


def create_grid(height, depth, period, current, ref_depth):
    '''
    '''
    np.save('data/height.npy', height)
    np.save('data/depth.npy', depth)
    np.save('data/period.npy', period)
    np.save('data/current.npy', current)
    np.save('data/ref_depth.npy', ref_depth)
    
    shape = (len(height), len(depth), len(period), len(current), len(ref_depth))

    flows = np.full(shape, np.nan)

    num = np.prod(shape)

    start = datetime.now()

    for i, ((ih, h), (id, d), (it, t), (iu, u)) in enumerate(enum_product(height, depth, period, current)):
        if h == 0 or d == 0 or t == 0:
            flows[ih, id, it, iu, :] = np.nan
            continue
        
        if ((i + 1) % 10 == 0):
            delta = (datetime.now() - start).total_seconds()
            rate = delta / i
            remaining = (num - i) * rate / 60
            print(f'{i / num * 100:.2f}% ({remaining:.2f} minutes remaining)', end='\r')
        
        try:
            w = _wavelength(h, t, d, u)
            f = raschii.FentonWave(h, d, w, N)
            flow = f.velocity(np.full_like(ref_depth, 0), ref_depth)[:, 0] + u
            flows[ih, id, it, iu, :] = flow

        except KeyboardInterrupt:
            exit()

        except:
            flows[ih, id, it, iu, :] = np.nan
    
    np.save('data/flows.npy', flows)


def _dispersion_stokes(k, h, t, d, u=0, g=G):
    '''
    Reference
    ---------
    On calculating the lengths of water waves, Fenton and McKee (1990), Equation (1)
    Nonlinear Wave Theories, Fenton (1990), Table 1
    '''
    S = 1 / np.cosh(2 * k * d)
    C0 = np.sqrt(np.tanh(k * d))
    C2 = C0 * (2 + 7 * S**2) / (4 * (1 - S)**2)
    C4 = C0 * (4 + 32*S - 116*S**2 - 400*S**3 - 71*S**4 + 146*S**5) / (32 * (1 - S)**5)

    return (
        np.sqrt(k / g) * u - 2 * np.pi / t / np.sqrt(g * k) + C0 + (k * h / 2)**2 * C2 + \
        (k * h / 2)**4 * C4
    )


def _wavelength(height, period, depth, u=0, g=G):
    '''
    Reference
    ---------
    On calculating the lengths of water waves, Fenton and McKee (1990), Abstract
    '''
    l0 = G * period ** 2 / 2 / np.pi * np.tanh((2 * np.pi * np.sqrt(depth / g) / period) ** 1.5) ** (2/3)
    k0 = 2 * np.pi / l0

    k = optimize.newton(lambda k: _dispersion_stokes(k, height, period, depth, u, g), k0)

    return 2 * np.pi / k


def fill_grid():
    flow = np.load('data/flows.npy')
    flow[0, :, :, :, :] = 0
    flow[:, 0, :, :, :] = 0
    flow[:, :, 0, :, :] = 0

    valid_mask = ~np.isnan(flow)
    valid_coords = np.array(np.nonzero(valid_mask)).T 
    valid_values = flow[valid_mask] 

    nan_coords = np.array(np.nonzero(~valid_mask)).T 

    flow[tuple(nan_coords.T)] = interpolate.griddata(valid_coords, valid_values, nan_coords, method='linear')

    np.save('data/flows_filled3.npy', flow)


if __name__ == '__main__':
    height    = np.arange(0, 28, 2)
    depth     = np.arange(0, 210, 20)
    period    = np.arange(21)
    current   = np.arange(6)
    ref_depth = np.arange(0, 0.35, 0.05)

    # fill_grid()
    # exit()

    # create_grid(height, depth, period, current, ref_depth)
    # exit()

    flow_interp = get_interpolator(filled=True)

    current = 0.03
    ref_depth = 0.2

    height = np.linspace(0, 18, num=12)
    depth = [20, 50, 100]
    period = [0, 10, 15, 18]

    import matplotlib.cm as cm
    colors = cm.get_cmap('tab20', np.size(depth) * np.size(period))

    for i, (d, t) in enumerate(itertools.product(depth, period)):
        grid_flows = []
        fourier_flows = []
        
        for h in height:
            grid_flow = flow_interp([h, d, t, current, ref_depth])[0]

            grid_flows.append(grid_flow)

            write_data(h, d, t, current, N)
            res = sp.run(['./Fourier'], cwd='./Fourier', capture_output=True, text=True).stdout
            flow = read_flow_data(d, ref_depth)
            if flow > 5: flow = np.nan
            fourier_flows.append(flow)

        plt.plot(height, grid_flows, label=f'Grid d={d} t={t}', color=colors(i))
        plt.plot(height, fourier_flows, label=f'Fourier d={d} t={t}', linestyle='dashed', color=colors(i))

    plt.legend()
    plt.show()

    # plot flow speeds from raschii and grid