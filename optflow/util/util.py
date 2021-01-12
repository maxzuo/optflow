import numpy as np

try:
    from tqdm import tqdm
except:
    from sys import stderr
    from os import environ
    # dummy tqdm object
    tqdm = lambda a, **k: a
    tqdm.update = lambda a: None
    if not environ.get('optflow_warning', None) == 'SUPPRESS':
        print("Note: tqdm not installed. Install tqdm module to display progress bars.", file=stderr)


def dist(x, c):
    """ Calculates the pairwise Euclidian distance between every row in x and every row in c.

    Inputs:
    - x: NxD matrix where each row represents a feature vector
    - c: MxD matrix where each row represents a feature vector

    Outputs:
    - d: NxM where the value at (i, j) represents the Euclidian distance between features x_i and c_j
    """
    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise NameError("Data dimension does not match dimension of centers")

    n2 = np.transpose(
            np.dot(
                np.ones((ncentres, 1)),
                np.transpose(np.sum(np.square(x), 1).reshape(ndata, 1)),
            )
        ) + np.dot(
            np.ones((ndata, 1)),
            np.transpose(np.sum(np.square(c), 1).reshape(ncentres, 1)),
        ) - 2 * np.dot(x, np.transpose(c))

    n2[n2 < 0] = 0
    return n2