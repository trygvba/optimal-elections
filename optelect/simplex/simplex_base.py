"""Module for basic simplex algorithm."""
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .matrix_operations import pivot


class SimplexBase:
    """Class for basic simplex algorithm.

    Solves basic linear programs of the form
    maximize c^T * x, under the constraints
    A * x <= b, and x >= 0.
    """

    weight_tol: float = 1e-8
    max_iterations: int = 1000

    def __init__(self, mat: ArrayLike, bvec: ArrayLike, cvec: ArrayLike):
        """Create a SimplexBase object.

        Args:
            mat: Matrix A that specifies upper constraints.
            bvec: Threshold values for upper constraints.
            cvec: Array specifying linear objective functional.
        """
        self._mat = np.array(mat)
        self._b = np.array(bvec)
        assert self._mat.shape[0] == len(
            self._b
        ), "Length of b must equal number of constraints."
        self._c = np.array(cvec)
        assert self._mat.shape[1] == len(
            self._c
        ), "Length of c must equal number of columns in matrix."
        self.tableau = self._assemble_tableau()

        # For now we'll just assume our initial BFS is the corner
        # at the vertex.
        self._basic_vars = np.arange(
            start=self.num_unknowns,
            stop=self.num_unknowns + self.num_constraints,
            dtype=int,
        )

    @property
    def num_unknowns(self) -> int:
        """Number of unknowns in linear program."""
        return self._mat.shape[1]

    @property
    def num_constraints(self) -> int:
        """Number of constraints in linear program."""
        return self._mat.shape[0]

    @property
    def current_basic_variables(self) -> NDArray:
        """Get current basic variables."""
        return self._basic_vars

    @property
    def current_solution(self) -> NDArray:
        """Get current basic feasible solution."""
        res = np.zeros(self.num_constraints + self.num_unknowns)
        res[self._basic_vars] = self.tableau[1:, -1]
        return res[: self.num_unknowns]

    def _assemble_tableau(self) -> NDArray:
        """Create the standard tableay for linear program."""
        tab = np.zeros(
            (1 + self.num_constraints, 2 + self.num_constraints + self.num_unknowns),
            dtype=float,
        )
        # Insert initial values:
        tab[0, 0] = 1.0
        tab[0, 1 : (1 + self.num_unknowns)] = -self._c
        tab[1:, 1 : (1 + self.num_unknowns)] = self._mat
        tab[
            1:, (1 + self.num_unknowns) : (1 + self.num_unknowns + self.num_constraints)
        ] = np.eye(self.num_constraints, dtype=float)
        tab[1:, -1] = self._b.T
        return tab

    def get_pivot_column(self) -> Optional[int]:
        """Returns the best candidate for pivot column.

        Based on current form of the program tableay.

        Returns:
            Index of the column to pivot.
            Index is relative to tableau.
            If None, then no column will yield a better
            basic feasible then what you currently have.
        """
        obj_diff = self.tableau[0, 1 : (self.num_constraints + self.num_unknowns)]
        # Check if no weights are positive:
        if np.all(obj_diff < self.weight_tol):
            return None
        return 1 + np.argmax(obj_diff, axis=0)

    def get_pivot_row(self, col: int) -> int:
        """Returns the best candidate for pivot row.

        Based on current form of the program tableau.

        Args:
            col: Index of column to pivot against.

        Returns:
            Index of the row to pivot
        """
        # TODO: Also need to make sure that all variables will remain positive.
        return 1 + np.argmin(self.tableau[1:, -1] / self.tableau[1:, col])

    @staticmethod
    def local_index(index: int) -> int:
        """Get unknowns index from tableau index."""
        return index - 1

    def iterate(self) -> bool:
        """Make one iteration of the simplex algorithm.

        Based on current state of program.

        Returns:
            True if solution has already been reached (no change)
            False, otherwise.
        """
        # First, we find the pivot column
        if pcol := self.get_pivot_column():
            # Then get pivot row
            prow = self.get_pivot_row(col=pcol)
            # We should pivot, god damn it
            self.tableau = pivot(mat=self.tableau, row=prow, col=pcol)
            # Finally, exchange the basic variables.
            old_basic = self._basic_vars[self.local_index(prow)]
            new_basic = self.local_index(pcol)
            self._basic_vars[self._basic_vars == old_basic] = new_basic
            return False
        return True
