import numpy as np
import subprocess as sp
import itertools
from scipy import interpolate
from scipy import optimize
import os
import raschii
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle

from utils import *
from fourier import *


J = os.path.join


def save_interpolator(dir):
    '''
    '''
    height = np.load(J(dir, 'height.npy'))
    depth = np.load(J(dir, 'depth.npy'))
    period = np.load(J(dir, 'period.npy'))
    current = np.load(J(dir, 'current.npy'))
    ref_depth = np.load(J(dir, 'ref_depth.npy'))

    headers = (height, depth, period, current, ref_depth)

    flows = np.load(J(dir, 'flows.npy'))
    flows = flows.ravel()

    coords = np.array(np.meshgrid(*headers)).T.reshape(-1, len(headers))

    idxs = ~np.isnan(flows)
    coords = coords[idxs]
    flows = flows[idxs]

    print('Initialising interpolator...')

    interp = MyInterp(coords, flows)

    with open(J(dir, 'interp.pkl'), 'wb') as f:
        pickle.dump(interp, f)

    print(f'Saved {dir}.')


def create_grid(height, depth, period, current, ref_depth, method='raschii', dir='.'):
    '''
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

    m = 'raschii' if 'raschii'.startswith(method.strip().lower()) else 'fourier'

    np.save(f'{dir}/height_{m}.npy', height)
    np.save(f'{dir}/depth_{m}.npy', depth)
    np.save(f'{dir}/period_{m}.npy', period)
    np.save(f'{dir}/current_{m}.npy', current)
    np.save(f'{dir}/ref_depth_{m}.npy', ref_depth)
    
    shape = (len(height), len(depth), len(period), len(current), len(ref_depth))
    flows = np.full(shape, np.nan)
    num = np.prod(shape)
    start = datetime.now()

    for i, ((ih, h), (id, d), (it, t), (iu, u)) in enumerate(enum_product(height, depth, period, current)):
        if h == 0 or d == 0 or t == 0:
            continue
        
        if ((i + 1) % 10 == 0):
            delta = (datetime.now() - start).total_seconds()
            rate = delta / i
            remaining = (num - i) * rate / 60
            print(f'{i / num * 100:.2f}% ({remaining:.2f} minutes remaining)', end='\r')
        
        try:
            if 'raschii'.startswith(method):
                w = _wavelength(h, t, d, u)
                f = raschii.FentonWave(h, d, w, N)
                flow = f.velocity(np.full_like(ref_depth, 0), ref_depth)[:, 0] + u

            else:
                write_data(h, d, t, u, N)
                res = sp.run(['./Fourier'], cwd='./Fourier', capture_output=True, text=True).stdout
                
                if 'not converged' in res: flow = np.nan
                else: flow = read_flow_data(d, ref_depth)

            flows[ih, id, it, iu, :] = flow

        except KeyboardInterrupt:
            exit()

        except:
            flows[ih, id, it, iu, :] = np.nan
    
    np.save(f'{dir}/flows_{m}.npy', flows)


def _dispersion_stokes(k, h, t, d, u=0, g=G):
    '''
    Reference
    ---------
    On calculating the lengths of water waves, Fenton and McKee (1990), Equation 1
    Nonlinear Wave Theories, Fenton (1990), Table 1
    '''
    S = 1 / np.cosh(2 * k * d)
    C0 = np.sqrt(np.tanh(k * d))
    C2 = C0 * (2 + 7 * S**2) / (4 * (1 - S)**2)
    C4 = C0 * (4 + 32*S - 116*S**2 - 400*S**3 - 71*S**4 + 146*S**5) / (32 * (1 - S)**5)

    return (np.sqrt(k / g) * u - 2 * np.pi / t / np.sqrt(g * k) + C0 + (k * h / 2)**2 * C2 + (k * h / 2)**4 * C4)


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


def breaking_limit(wavelength, depth):
    '''
    Fenton Nonlinear Wave Theories (1990) Equation 32
    '''
    w_d = wavelength / depth
    return depth * (0.141063 * w_d +  0.0095721 * w_d ** 2 + 0.0077829 * w_d ** 3) / (1 + 0.0788340 * w_d + 0.0317567 * w_d ** 2 + 0.0093407 * w_d ** 3)


MAX_FLOW = 11


def fill_grid(input, output):
    '''
    '''
    flow = np.load(input)

    # assumes headers start from 0
    flow[0, :, :, :, :] = 0
    flow[:, 0, :, :, :] = 0
    flow[:, :, 0, :, :] = 0

    # remove outliers
    flow[flow < 0] = np.nan
    flow[flow > MAX_FLOW] = np.nan

    valid_mask = ~np.isnan(flow)
    valid_coords = np.array(np.nonzero(valid_mask)).T 
    valid_values = flow[valid_mask] 
    nan_coords = np.array(np.nonzero(~valid_mask)).T

    flow[tuple(nan_coords.T)] = interpolate.griddata(valid_coords, valid_values, nan_coords, method='linear')

    np.save(output, flow)


def test_interp(interp):
    '''
    '''
    current = 0.03
    ref_depth = 0.2

    height = np.arange(0, 29, 0.72)
    depth = [20, 100, 200]
    period = [5, 15, 21]

    import matplotlib.cm as cm
    colors = cm.get_cmap('tab20', np.size(depth) * np.size(period))

    for i, (d, t) in enumerate(itertools.product(depth, period)):
        grid_flows = []
        fourier_flows = []
        hms = []
        
        for h in height:
            print('Interpolating... ', end='', flush=True)
            grid_flow = interp([h, d, t, current, ref_depth])[0]
            print('Done.')

            grid_flows.append(grid_flow)

            print('Fouriering... ', end='', flush=True)
            write_data(h, d, t, current, N)
            res = sp.run(['./Fourier'], cwd='./Fourier', capture_output=True, text=True).stdout
            flow = read_flow_data(d, ref_depth)
            print('Done')
            
            # if flow > 5: flow = np.nan
            fourier_flows.append(flow)

            try:
                hm = breaking_limit(_wavelength(h, t, d, current), d)
                hms.append(hm)
                
            except ZeroDivisionError: pass

        plt.axvline(np.mean(hms), color=colors(i), linestyle='dotted')
        plt.plot(height, grid_flows, label=f'Grid d={d} t={t}', color=colors(i))
        plt.plot(height, fourier_flows, label=f'Fourier d={d} t={t}', linestyle='dashed', color=colors(i), linewidth=2)

    plt.xlim([0, 40])
    plt.ylim([0, 6])
    plt.legend()
    plt.show()

    # plot flow speeds from raschii and grid


if __name__ == '__main__':
    height    = np.arange(0, 28, 2)
    depth     = np.arange(0, 210, 20)
    period    = np.arange(21)
    current   = np.arange(6)
    ref_depth = np.arange(0, 0.55, 0.05)

    # create_grid(height, depth, period, current, ref_depth, method='fourier', dir='fourier_grid')
    # create_grid(height, depth, period, current, ref_depth, method='raschii', dir='raschii_grid')
    
    fill_grid('fourier_grid/flows.npy', 'fourier_grid/flows_filled.npy')
    fill_grid('raschii_grid/flows.npy', 'raschii_grid/flows_filled.npy')

    # save_interpolator(dir='raschii_grid')
    # save_interpolator(dir='fourier_grid')
    # exit()

    # with open('raschii_grid/interp.pkl', 'rb') as f:
    #     interp = pickle.load(f)


    
    test_interp(interp)

