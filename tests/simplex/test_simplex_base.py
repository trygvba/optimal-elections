import numpy as np
import pytest

from optelect.simplex.simplex_base import SimplexBase


@pytest.fixture
def linprob() -> SimplexBase:
    A = np.array([[3.0, 2.0, 1], [2, 5, 3]])
    b = np.array([10, 15])
    c = np.array([-2, -3, -4])
    return SimplexBase(mat=A, bvec=b, cvec=c)


def test_simplex_base(linprob):
    # Assert
    assert linprob.num_constraints == 2
    assert linprob.num_unknowns == 3
    assert np.allclose(
        linprob.tableau,
        np.array(
            [[1, 2, 3, 4, 0, 0, 0], [0, 3, 2, 1, 1, 0, 10], [0, 2, 5, 3, 0, 1, 15]]
        ),
    )


def test_initial_basic_solution(linprob):
    assert np.all(np.array([3, 4]) == linprob.current_basic_variables)
    assert np.allclose(np.zeros(3), linprob.current_solution)


def test_first_pivot(linprob):
    pcol = linprob.get_pivot_column()
    assert pcol == 3
    assert linprob.get_pivot_row(pcol) == 2


def test_first_iteration(linprob):
    assert not linprob.iterate()
    assert np.all(np.array([3, 2]) == linprob.current_basic_variables)
    assert np.allclose(np.array([0.0, 0.0, 5.0]), linprob.current_solution)
