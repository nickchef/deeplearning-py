import numpy as np


def xavier_normal_init(in_dim: int, out_dim: int) -> tuple:
    """
    Xavier normal initialization.

    w = N(0, 2/(in_dim+out_dim)), b = 0

    Parameters
    ----------
    in_dim:
        input dimension
    out_dim:
        output dimension

    Returns
    -------
    out:
        tuple of initialized weight ndarray and bias ndarray
    """
    return np.random.randn(out_dim, in_dim) * 2. / (out_dim+in_dim), np.zeros((out_dim, 1))


def xavier_uniform_init(in_dim: int, out_dim: int) -> tuple:
    """
    Xavier uniform initialization.

    w = u(-sqrt(6/(in_dim+out_dim)), sqrt(6/(in_dim+out_dim))), b = 0

    Parameters
    ----------
    in_dim:
        input dimension
    out_dim:
        output dimension

    Returns
    -------
    out:
        tuple of initialized weight ndarray and bias ndarray
    """
    return np.random.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                             high=np.sqrt(6. / (in_dim + out_dim)),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_uniform_init(in_dim, out_dim):
    """
    Kaiming uniform initialization

    w = u(-sqrt(6/out_dim), sqrt(6/out_dim)), b = 0

    Parameters
    ----------
    in_dim:
        input dimension
    out_dim:
        output dimension

    Returns
    -------
    out:
        tuple of initialized weight ndarray and bias ndarray
    """
    return np.random.uniform(low=-np.sqrt(6. / out_dim),
                             high=np.sqrt(6. / out_dim),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_normal_init(in_dim, out_dim):
    """
    Kaiming normal initialization.

    w = N(0, 2/out_dim), b = 0

    Parameters
    ----------
    in_dim:
        input dimension
    out_dim:
        output dimension

    Returns
    -------
    out:
        tuple of initialized weight ndarray and bias ndarray
    """
    return np.random.randn(out_dim, in_dim) * 2. / out_dim, np.zeros((out_dim, 1))

