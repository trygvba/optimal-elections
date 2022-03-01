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


def test_row_combine():
    # Arrange
    mat = np.arange(9, dtype=float).reshape((3, 3))
    # Act
    res = sut.row_combine(mat=mat, row1=0, row2=1, factor=-0.5)
    # Assert
    assert np.allclose(
        np.array([[-1.5, -1.0, -0.5], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]), res
    )


def test_pivot():
    # Arrange
    mat = np.arange(9, dtype=float).reshape((3, 3))
    # Act
    res = sut.pivot(mat, row=1, col=1)
    # Assert
    np.allclose(
        np.array(
            [
                [0.0 - 1.0 * 0.75, 0.0, 2.0 - 1.0 * 1.25],
                [0.75, 1.0, 1.25],
                [6.0 - 7 * 0.75, 0.0, 8.0 - 7 * 1.25],
            ]
        ),
        res,
    )
