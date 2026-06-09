import numpy as np
import scipy
import logging

from .utils import eliminate_zeros


class CompositeBase:
    xp = np
    sparse_lib = scipy.sparse

    def __init__(self, bases):
        assert isinstance(bases, (list, tuple)), (
            f"You have to supply the bases for a composite base as tuple or list but gave {type(bases)}"
        )
        self.bases = bases
        self.logger = logging.getLogger(name="Spectral Discretization")

    @property
    def ndim(self):
        return len(self.bases)

    def get_grid(self):
        """
        Get grid in physical space
        """
        grids = [self.bases[i].get_grid() for i in range(self.ndim)]
        return self.xp.meshgrid(*grids, indexing="ij")

    def get_wavenumbers(self):
        """
        Get grid in spectral space
        """
        grids = [self.bases[i].get_wavenumbers() for i in range(self.ndim)]
        return self.xp.meshgrid(*grids, indexing="ij")

    def transform(self, u, axes=None):
        axes = axes if axes else tuple(i for i in range(self.ndim))

        u_hat = u.copy()
        for axis in axes:
            u_hat = self.bases[axis].transform(u_hat, axes=(axis,))

        return u_hat

    def itransform(self, u_hat, axes=None):
        axes = axes if axes else tuple(i for i in range(self.ndim))

        u = u_hat.copy()
        for axis in axes:
            u = self.bases[axis].itransform(u, axes=(axis,))

        return u

    def expand_matrix_ND(self, matrix, aligned):
        sp = self.sparse_lib
        axes = np.delete(np.arange(self.ndim), aligned)
        ndim = len(axes) + 1

        if ndim == 1:
            mat = matrix
        elif ndim == 2:
            axis = axes[0]
            I1D = sp.eye(self.bases[axis].N)

            mats = [None] * ndim
            mats[aligned] = matrix
            mats[axis] = I1D

            mat = sp.kron(*mats)
        elif ndim == 3:
            mats = [None] * ndim
            mats[aligned] = matrix
            for axis in axes:
                I1D = sp.eye(self.bases[axis].N)
                mats[axis] = I1D

            mat = sp.kron(mats[0], sp.kron(*mats[1:]))

        else:
            raise NotImplementedError(
                f"Matrix expansion not implemented for {ndim} dimensions!"
            )

        mat = eliminate_zeros(self.sparse_lib, mat)
        return mat

    def get_differentiation_matrix(self, axes, **kwargs):
        """
        Get differentiation matrix along specified axis. `kwargs` are forwarded to the 1D base implementation.

        Args:
            axes (tuple): Axes along which to differentiate.

        Returns:
            sparse differentiation matrix
        """
        D = self.expand_matrix_ND(
            self.bases[axes[0]].get_differentiation_matrix(**kwargs), axes[0]
        )
        for axis in axes[1:]:
            _D = self.bases[axis].get_differentiation_matrix(**kwargs)
            D = D @ self.expand_matrix_ND(_D, axis)

        self.logger.debug(
            f"Set up differentiation matrix along axes {axes} with kwargs {kwargs}"
        )
        return D
