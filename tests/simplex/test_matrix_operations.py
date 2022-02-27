import numpy as np
import pytest

from optelect.simplex import matrix_operations as sut


def test_scale_row_good_input():
    # Arrange
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Act
    result = sut.scale_row(mat, 0, 0.5)
    # Assert
    assert np.allclose(np.array([[0.5, 1.0], [3.0, 4.0]]), result)


def test_scale_row_not_matrix():
    # Arrange
    notmat = np.arange(27).reshape((3, 3, 3))
    # Act
    with pytest.raises(AssertionError):
        notmat = sut.scale_row(notmat, 0, 3.0)


def test_scale_row_too_big_row():
    # Arrange
    mat = np.arange(4).reshape((2, 2))
    # Act
    with pytest.raises(AssertionError):
        mat = sut.scale_row(mat, 2, -0.5)
