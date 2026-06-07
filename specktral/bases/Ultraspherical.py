import numpy as np
import scipy

from ..utils import cache
from .Chebychev import Chebychev

from scipy.special import factorial
from functools import partial


class Ultraspherical(Chebychev):
    """
    This implementation follows https://doi.org/10.1137/120865458.
    The ultraspherical method works in Chebychev polynomials as well, but also uses various Gegenbauer polynomials.
    The idea is that for every derivative of Chebychev T polynomials, there is a basis of Gegenbauer polynomials where the differentiation matrix is a single off-diagonal.
    There are also conversion operators from one derivative basis to the next that are sparse.

    This basis is used like this: For every equation that you have, look for the highest derivative and bump all matrices to the correct basis. If your highest derivative is 2 and you have an identity, it needs to get bumped from 0 to 1 and from 1 to 2. If you have a first derivative as well, it needs to be bumped from 1 to 2.
    You don't need the same resulting basis in all equations. You just need to take care that you translate the right hand side to the correct basis as well.
    """

    def get_differentiation_matrix(self, p=1):
        """
        Notice that while sparse, this matrix is not diagonal, which means the inversion cannot be parallelized easily.

        Args:
            p (int): Order of the derivative

        Returns:
            sparse differentiation matrix
        """
        sp = self.sparse_lib
        xp = self.xp
        N = self.N
        l = p
        return 2 ** (l - 1) * factorial(l - 1) * sp.diags(xp.arange(N - l) + l, offsets=l) / self.lin_trf_fac**p

    def get_S(self, lmbda):
        """
        Get matrix for bumping the derivative base by one from lmbda to lmbda + 1. This is the same language as in
        https://doi.org/10.1137/120865458.

        Args:
            lmbda (int): Ingoing derivative base

        Returns:
            sparse matrix: Conversion from derivative base lmbda to lmbda + 1
        """
        N = self.N

        if lmbda == 0:
            sp = scipy.sparse
            mat = ((sp.eye(N) - sp.eye(N, k=2)) / 2.0).tolil()
            mat[:, 0] *= 2
        else:
            sp = self.sparse_lib
            xp = self.xp
            mat = sp.diags(lmbda / (lmbda + xp.arange(N))) - sp.diags(
                lmbda / (lmbda + 2 + xp.arange(N - 2)), offsets=+2
            )

        return self.sparse_lib.csc_matrix(mat)

    def get_basis_change_matrix(self, p_in=0, p_out=0, **kwargs):
        """
        Get a conversion matrix from derivative base `p_in` to `p_out`.

        Args:
            p_out (int): Resulting derivative base
            p_in (int): Ingoing derivative base
        """
        mat_fwd = self.sparse_lib.eye(self.N)
        for i in range(min([p_in, p_out]), max([p_in, p_out])):
            mat_fwd = self.get_S(i) @ mat_fwd

        if p_out > p_in:
            return mat_fwd

        else:
            # We have to invert the matrix on CPU because the GPU equivalent is not implemented in CuPy at the time of writing.
            import scipy.sparse as sp

            if self.useGPU:
                mat_fwd = mat_fwd.get()

            mat_bck = sp.linalg.inv(mat_fwd.tocsc())

            return self.sparse_lib.csc_matrix(mat_bck)

    def get_integration_matrix(self):
        """
        Get an integration matrix. Please use `UltrasphericalHelper.get_integration_constant` afterwards to compute the
        integration constant such that integration starts from x=-1.

        Example:

        .. code-block:: python

            import numpy as np
            from pySDC.helpers.spectral_helper import UltrasphericalHelper

            N = 4
            helper = UltrasphericalHelper(N)
            coeffs = np.random.random(N)
            coeffs[-1] = 0

            poly = np.polynomial.Chebyshev(coeffs)

            S = helper.get_integration_matrix()
            U_hat = S @ coeffs
            U_hat[0] = helper.get_integration_constant(U_hat, axis=-1)

            assert np.allclose(poly.integ(lbnd=-1).coef[:-1], U_hat)

        Returns:
            sparse integration matrix
        """
        return (
            self.sparse_lib.diags(1 / (self.xp.arange(self.N - 1) + 1), offsets=-1)
            @ self.get_basis_change_matrix(p_out=1, p_in=0)
            * self.lin_trf_fac
        )

    def get_integration_constant(self, u_hat, axis):
        """
        Get integration constant for lower bound of -1. See documentation of `UltrasphericalHelper.get_integration_matrix` for details.

        Args:
            u_hat: Solution in spectral space
            axis: Axis you want to integrate over

        Returns:
            Integration constant, has one less dimension than `u_hat`
        """
        slices = [
            None,
        ] * u_hat.ndim
        slices[axis] = slice(1, u_hat.shape[axis])
        return self.xp.sum(u_hat[(*slices,)] * (-1) ** (self.xp.arange(u_hat.shape[axis] - 1)), axis=axis)
