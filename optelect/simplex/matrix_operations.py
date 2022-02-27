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
