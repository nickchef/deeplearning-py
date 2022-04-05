import numpy as np


def xavier_normal_init(in_dim, out_dim):
    """
    Xavier normal initialization.

    Parameters
    ----------
    in_dim
    out_dim

    Returns
    -------

    """
    return np.random.randn(out_dim, in_dim) * 2. / (out_dim+in_dim), np.zeros((out_dim, 1))


def xavier_uniform_init(in_dim, out_dim):
    return np.random.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                             high=np.sqrt(6. / (in_dim + out_dim)),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_uniform_init(in_dim, out_dim):
    return np.random.uniform(low=-np.sqrt(6. / out_dim),
                             high=np.sqrt(6. / out_dim),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_normal_init(in_dim, out_dim):
    return np.random.randn(out_dim, in_dim) * 2. / out_dim, np.zeros((out_dim, 1))

