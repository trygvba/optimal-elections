import numpy as np

from optelect.simplex.simplex_base import SimplexBase


def test_simplex_base():
    # Arrange
    A = np.array([[3.0, 2.0, 1], [2, 5, 3]])
    b = np.array([10, 15])
    c = np.array([-2, -3, -4])
    linprob = SimplexBase(mat=A, bvec=b, cvec=c)
    # Assert
    assert linprob.num_constraints == 2
    assert linprob.num_unknowns == 3
    assert np.allclose(
        linprob.tableau,
        np.array(
            [[1, 2, 3, 4, 0, 0, 0], [0, 3, 2, 1, 1, 0, 10], [0, 2, 5, 3, 0, 1, 15]]
        ),
    )
