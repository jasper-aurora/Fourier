from scipy import interpolate
from scipy import spatial
import numpy as np
import itertools


# ----- Globals -----

G = 9.81
N = 22

# -------------------


class MyRegularGridInterp:
    '''
    Performs linear interpolation within the bounds and nearest neighbour interpolation outside of the bounds.
    '''
    def __init__(self, points, values, method='linear'):
        self.interp = interpolate.RegularGridInterpolator(points, values, method=method, bounds_error=False, fill_value=np.nan)
        self.nearest = interpolate.RegularGridInterpolator(points, values, method='nearest', bounds_error=False, fill_value=None)

    def __call__(self, xi):
        xi = np.asarray(xi)
        vals = self.interp(xi)
        idxs = np.isnan(vals)
        if idxs.any():
            if xi.ndim == 1: xi = xi[np.newaxis, ...]
            vals[idxs] = self.nearest(xi[idxs])
        return vals
    

class MyInterp:
    '''
    Performs linear interpolation within the bounds and nearest neighbour interpolation outside of the bounds.
    '''
    def __init__(self, points, values):
        delaunay = spatial.Delaunay(points)
        self.interp = interpolate.LinearNDInterpolator(delaunay, values)
        self.nearest = interpolate.NearestNDInterpolator(delaunay, values)

    def __call__(self, xi):
        xi = np.asarray(xi)
        vals = self.interp(xi)
        idxs = np.isnan(vals)
        if idxs.any():
            if xi.ndim == 1: xi = xi[np.newaxis, ...]
            vals[idxs] = self.nearest(xi[idxs])
        return vals


def str_after(s: str, sub_str: str) -> str:
    '''
    Returns
    -------
    x : str
        Next word after `sub_str` in `s`.
    '''
    i = s.index(sub_str) + len(sub_str)
    return s[i:].split()[0]


def enum_product(*args):
    '''
    Returns cartesian product of args, with each element prepended with its index.
    '''
    return itertools.product(*[enumerate(x) for x in args])


if __name__ == '__main__':
    from datetime import datetime

    n = datetime.now()
    i = 0
    while True:
        if (datetime.now() - n).total_seconds() >= 1:
            print(i)
            i += 1
            n = datetime.now()


    interp = MyInterp([[0, 1], [0, 1]], [[1, 2], [3, 4]])
    print(interp([[]]))