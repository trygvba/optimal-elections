"""Module with some useful matrix operations."""
from numpy.typing import NDArray


def scale_row(mat: NDArray, row: int, factor: float) -> NDArray:
    """Scale a row of a matrix.

    Args:
        mat: A matrix containing a row to scale.
        row: Index of row to scale
        factor: factor to scale row by.

    Returns:
        Matrix x, where x[row] has been scaled.

    Examples:
        >>> import numpy as np
        >>> x = np.array([[1., 2.],[3., 4.]])
        >>> scale_row(x, 0, 0.5)
        array([[0.5, 1. ],
               [3. , 4. ]])
    """
    assert len(mat.shape) == 2
    assert row < mat.shape[0]
    mat[row] *= factor
    return mat


def row_combine(mat: NDArray, row1: int, row2: int, factor: float) -> NDArray:
    """Combine rows of a matrix.

    Args:
        mat: Matrix to row combine on.
        row1: index of first row to combine.
        row2: index of second row to combine.
        factor: Scaling factor.

    Returns:
        mat, but where
        mat[row1] == mat[row1] + factor*mat[row2]
    """
    mat[row1] += factor * mat[row2]
    return mat


def pivot(mat: NDArray, row: int, col: int) -> NDArray:
    """Pivot matrix on given row and column.

    Args:
        mat: Matrix to pivot.
        row: index of row to pivot
        col: index of column to pivot.

    Returns:
        pivoted matrix.
    """
    assert abs(mat[row, col]) > 1e-8, "Cannot pivot around a 0 element of a matrix"
    # First scale the row so we have a one at the pivot element:
    mat = scale_row(mat, row=row, factor=1.0 / mat[row, col])
    for rid in range(mat.shape[0]):
        if rid != row:
            mat = row_combine(mat=mat, row1=rid, row2=row, factor=-mat[rid, col])
    return mat
