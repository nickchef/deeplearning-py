import numpy as np


def xavier_normal_init(in_dim, out_dim):
    return np.random.randn(out_dim, in_dim) * np.sqrt(6. / (out_dim+in_dim)), np.zeros((out_dim, 1))


def xavier_uniform_init(in_dim, out_dim):
    return np.random.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                             high=np.sqrt(6. / (in_dim + out_dim)),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_uniform_init(in_dim, out_dim):
    return np.random.uniform(low=-np.sqrt(2. / in_dim),
                             high=np.sqrt(2. / in_dim),
                             size=(out_dim, in_dim)), np.zeros((out_dim, 1))


def kai_ming_normal_init(in_dim, out_dim):
    return np.random.randn(out_dim, in_dim) * np.sqrt(2. / in_dim), np.zeros((out_dim, 1))

