import cvxpy as cp
import numpy as np


def reconstruct_1bit(y, A, verbose=True):
    """
    Solve the modified basis pursuit problem by Plan and Vershynin.

    Solves the linear program

        min ||x||_1
        s.t. sign(Ax) = y
         and ||Ax||_1 = m

    Using dummy variables for ||x||_1, can be removed since cvxpy handles
    them.

    Args:
        y: (m,) ndarray
        A: (m, n) ndarray
    """
    m, n = A.shape
    x = cp.Variable(shape=n)

    constraints = []

    # y_i*<a_i, x> >= 0
    constraints += [
            cp.multiply(y, (A @ x)) >= 0,
        ]

    # 1/m sum() >= 1
    constraints.append(cp.sum(cp.multiply(y, A @ x)) / m >= 1)

    objective = cp.Minimize(cp.norm1(x))
    prob = cp.Problem(objective, constraints)
    print("Calling solver.")
    prob.solve(verbose=verbose)

    return x.value


def reconstruct(y, A):
    """
    Solve the Basis pursuit problem with cvxpy.

    Solves the linear program

        min ||x||_1
        s.t. A @ x = y

    Args:
        y: (m,) ndarray
        A: (m, n) ndarray
    """
    m, n = A.shape
    x = cp.Variable(shape=n)

    constraints = [
        A @ x == y,
    ]

    objective = cp.Minimize(cp.norm1(x))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value
