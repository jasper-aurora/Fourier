import numpy as np
import subprocess as sp
import raschii
import matplotlib.pyplot as plt
import os
from Fenton import Fenton
from datetime import datetime
from scipy import optimize, special


DATA_FILE = 'Data.dat'
FLOW_FILE = 'Flowfield.res'

G = 9.81


def write_data(height, depth, length_str, length, N, current_crit=1, current_mag=0, height_steps=1):
    '''
    '''
    s = f'JP Test Wave\n' \
    f'{height / depth}  H/d\n' \
    f'{length_str}  Measure of length: "Wavelength" or "Period"\n' \
    f'{length / depth if length_str == "Wavelength" else length * np.sqrt(G / depth)}  Value of that length: L/d or T(g/d)^1/2 respectively\n' \
    f'{current_crit}  Current criterion (1 or 2)\n' \
    f'{current_mag / np.sqrt(G * depth)}  Current magnitude, (dimensionless) ubar/(gd)^1/2\n' \
    f'{N} Number of Fourier components or Order of Stokes/cnoidal theory\n' \
    f'{height_steps} Number of height steps to reach H/d\n' \
    'FINISH\n'

    # print(f'H/d={height / depth:.2f} T(g/d)^.5={length * np.sqrt(G / depth)}')
    
    with open(DATA_FILE, 'w') as f: f.write(s)


def read_flow_data(depth):
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

    return res


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


def _dispersion_cnoidal(m, h, t, d, u=0, g=G):
    '''
    On calculating the lengths of water waves, Fenton and McKee (1990), Equation 3 (https://doi.org/10.1016/0378-3839(90)90032-R)
    '''
    km = special.ellipk(m)
    em = special.ellipe(m)
    return u / np.sqrt(g / d) + 1 + h / m / d * (1 - 3 * em / 2 / km - m / 2) - np.sqrt(m * d**2 / 3 / g / h / t**2) * 4 * km


def _wavelength(height, period, depth, u=0, g=G):
    '''
    Reference
    ---------
    On calculating the lengths of water waves, Fenton and McKee (1990), Abstract
    '''
    L0 = G * period ** 2 / 2 / np.pi * np.tanh((2 * np.pi * np.sqrt(depth / G) / period) ** 1.5) ** (2/3)
    k0 = 2 * np.pi / L0

    print(L0 / depth)

    func = _dispersion_stokes if L0 / depth < 10 else _dispersion_cnoidal
    k = optimize.newton(lambda k: func(k, height, period, depth, u, g), k0)

    return 2 * np.pi / k


# def _wavelength(h, t, d, u=0, g=G):
#     a = 4 * np.pi**2 * d / g / t**2
#     b = a * np.sqrt(1 / np.tanh(a))
#     kd = (a + b**2 / np.cosh(b)**2) / (np.tanh(b) + b / np.cosh(b) ** 2)
#     return 2 * np.pi / kd * d


if __name__ == '__main__':
    height = np.arange(1, 21) # [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    depth = np.full_like(height, 30)
    period = np.full_like(height, 10)

    current = np.full_like(height, 1, dtype=np.float64)

    length_str = ['Period' for _ in range(len(height))]
    N = np.full_like(height, 22)
    steps = np.full_like(height, 1)
    relax = np.full_like(height, 0.5, dtype=np.float64)

    fourier_ws = []
    raschii_ws = []

    for height, depth, length_str, length, u, N, steps, relax in zip(height, depth, length_str, period, current, N, steps, relax):
        title = f'H: {height} D: {depth} T: {length} U: {u}' # N: {N} S: {steps} R: {relax}'
        print(f'\n{title}')
        
        t0 = datetime.now()

        write_data(height, depth, length_str, length, N, current_mag=u)

        res = sp.run(['./Fourier'], capture_output=True, text=True).stdout

        lines = [l.strip() for l in res.split('\n')]
        fourier_converged = True
        for l in lines:
            if 'not converged' in l:
                print(f'(Fourier) {l}')
                fourier_converged = False

        if fourier_converged:
            fourier_wavelength = [l for l in lines if 'Length/Depth' in l][0]
            fourier_wavelength = float(fourier_wavelength.split()[-1]) * depth

            flow = read_flow_data(depth)

            z = flow[:flow.shape[0] // 2, 0]
            fourier_vels = flow[:flow.shape[0] // 2, 1]

        else:
            z = np.arange(int(depth) // 2)
            fourier_vels = np.full_like(z, np.nan, dtype=np.float64)
            fourier_wavelength = 0

        t1 = datetime.now()
        fourier_time = (t1 - t0).total_seconds()

        try:
            x = np.full_like(z, 0)
            wavelength = _wavelength(height, length, depth, u)

            print(f'Wavelength F: {fourier_wavelength:.2f} R: {wavelength:.2f}')
            fourier_ws.append(fourier_wavelength)
            raschii_ws.append(wavelength)
            
            # continue

            f = raschii.FentonWave(height, depth, wavelength, N, relax=relax)
            raschii_vels = f.velocity(x, z)[:, 0]
            raschii_vels += u
            raschii_converged = True

        except Exception as e:
            print(f'(Raschii) {e}')
            raschii_vels = np.full_like(z, np.nan) # np.zeros_like(fourier_vels)
            raschii_converged = False

        t2 = datetime.now()
        raschii_time = (t2 - t1).total_seconds()

        # try:
        #     fenton_vels = np.empty_like(fourier_vels, dtype=np.float64)
        #     for i, zi in enumerate(z):
        #         fenton_vels[i] = Fenton(length, height, depth, u, 1, zi * 2, 0, 1).run()[0]
        #     fenton_converged = True
            
        # except Exception as e:
        #     print(f'(Fenton) {e}')
        #     fenton_vels = np.full_like(fourier_vels, np.nan)
        #     fenton_converged = False

        fenton_time = (datetime.now() - t2).total_seconds()

        print(f'Fourier: {fourier_time:.2f} Raschii: {raschii_time:.2f} Fenton: {fenton_time:.2f}')

        plt.figure()

        fourier_label = f'Fourier ({fourier_time:.2f}){" [Not converged]" if not fourier_converged else ""}'
        raschii_label = f'Raschii ({raschii_time:.2f}){" [Not converged]" if not raschii_converged else ""}'
        # fenton_label = f'Fenton ({fenton_time:.2f}){" [Not converged]" if not fenton_converged else ""}'
        
        plt.scatter(z, fourier_vels, label=fourier_label, color='red', alpha=0.5, marker='x')
        plt.scatter(z, raschii_vels, label=raschii_label, color='blue', alpha=0.5, marker='o')
        # plt.scatter(z, fenton_vels, label=fenton_label, color='green', alpha=0.5, marker='^')

        plt.xlabel('Depth (m)')
        plt.ylabel('Speed (m/s)')
        plt.title(title)
        plt.legend()

        plt.savefig(f'figs_new/{title}.png')
        plt.close()

    xs = np.arange(len(fourier_ws))
    plt.scatter(xs, fourier_ws, color='blue', label='Fourier', alpha=0.5, marker='x')
    plt.scatter(xs, raschii_ws, color='red', label='Raschii', alpha=0.5, marker='^')
    plt.legend()
    plt.show()

        

'''
T 0 - 20s,  main   5 - 15s
H 0 - 25m,  main  10 - 15m
D 0 - 200m, main  15 - 50m
U 0 - 5m/s, main 0.2 - 1m/s

R 0 - 0.5m, 0, 0.1, 0.2, 0.3, 0.4, 0.5
'''
