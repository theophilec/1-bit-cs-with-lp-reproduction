import cvxpy as cp
import numpy as np


def reconstruct_1bit(y, A):
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
    u = cp.Variable(shape=n)

    constraints = []
    # - u_i <= x_i <= u_i
    for i in range(n):
        constraints += [
            u[i] >= 0,
            x[i] >= -u[i],
            x[i] <= u[i],
        ]

    # y_i*<a_i, x> >= 0
    for i in range(m):
        constraints += [
            y[i] * cp.sum(cp.multiply(A[i], x)) >= 0,
        ]

    # 1/m sum() >= 1
    constraints.append(cp.sum(cp.multiply(y, A @ x)) / m >= 1)

    objective = cp.Minimize(cp.sum(u))
    prob = cp.Problem(objective, constraints)
    prob.solve()

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
