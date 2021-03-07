import numpy as np
import matplotlib.pyplot as plt


def sign(v):
    return np.where(v < 0, -1.0, 1.0)


def show_unit_ball(img):
    plt.imshow(img.reshape(28, 28) / np.linalg.norm(img))
    plt.show()


def generate_sparse_randn(n, s):
    x = np.zeros(n)
    I = np.random.randint(0, n, s)
    x[I] = np.random.randn(s)
    return x


def generate_sparse(n, s):
    x = np.zeros(n)
    I = np.random.randint(0, n, s)
    x[I] = 1
    return x
