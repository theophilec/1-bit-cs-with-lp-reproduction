import numpy as np
from numpy import pi as PI
from numpy.linalg import norm as norm2
import matplotlib.pyplot as plt


def sign(v):
    """
    Compute component-wise sign of vector.

    Zero handling: zero is positive by convention.

    Params:
        v: ndarray
    """
    return np.where(v < 0, -1.0, 1.0)


def img_show_unit_ball(img):
    """
    Show image reshaping as square and normalizing on unit sphere.

    Params:
        img: (n,) ndarray
    """
    side = int(np.sqrt(img.shape[0]))
    plt.imshow(img.reshape(side, side) / np.linalg.norm(img))
    plt.show()


def generate_sparse_randn(n, s):
    """
    Generate sparse signal with Normal components.

    Signal is of length n, with sparsity s.
    Components are iid Normal.

    Params:
        n: int, length of signal
        s: int, sparsity level. 0-norm of signal.
    """
    x = np.zeros(n)
    I = np.random.randint(0, n, s)
    x[I] = np.random.randn(s)
    return x


def generate_sparse(n, s):
    """
    Generate sparse signal with 1 values.

    Signal is of length n, with sparsity s.
    Components are equal to 1.

    Params:
        n: int, length of signal
        s: int, sparsity level. 0-norm of signal.
    """
    x = np.zeros(n)
    I = np.random.randint(0, n, s)
    x[I] = 1
    return x

def angular_error(v_est, v_true):
    """
    Compute angular error as in [18] (Section 3.1).

    Quantity is defined as:
        1 / pi arccos(<v_est, v_true>)

    Params:
        v_est: (n,) ndarray
        v_true: (n,) ndarray

    Note: v_est and v_true must have same size.
    """
    assert(v_est.shape == v_true.shape)

    return 1 / PI * np.arccos(np.sum(v_est * v_true))


def hamming_distance(v_est, v_true):
    """
    Compute Hamming distance as in [18] (Section 3.1).

    Quantity is defined as:
        1 / n * sum XOR(v_est[i], v_true[i])

    Params:
        v_est: (n,) ndarray
        v_true: (n,) ndarray

    Note: v_est and v_true must have same size.
    """
    assert(v_est.shape == v_true.shape)

    return 1 / len(v_est) * np.sum(v_est != v_true)

def snr(v_est, v_true):
    """
    Compute SNR as in [18] (p. 17).

    Quantity is defined as:
        10 log_10(||v_true||_2^2 / ||v_est - v_true||_2^2)

    Params:
        v_est: (n,) ndarray
        v_true: (n,) ndarray

    Note: v_est and v_true must have same size.
    """
    assert(v_est.shape == v_true.shape)

    return 10 * np.log10(norm2(v_true) ** 2 / norm2(v_est - v_true) ** 2)
