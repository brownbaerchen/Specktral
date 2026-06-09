import numpy as np

from ..utils import cache
from .core import SpectralOneDBase


class Chebychev(SpectralOneDBase):
    """
    The Chebychev base consists of special kinds of polynomials, with the main advantage that you can easily transform
    between physical and spectral space by discrete cosine transform.
    The differentiation in the Chebychev T base is dense, but can be preconditioned to yield a differentiation operator
    that moves to Chebychev U basis during differentiation, which is sparse. When using this technique, problems need to
    be formulated in first order formulation.

    This implementation is largely based on the Dedalus paper (https://doi.org/10.1103/PhysRevResearch.2.023068).
    """

    def __init__(self, *args, x0=-1, x1=1, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            x0 (float): Coordinate of left boundary. Note that only -1 is currently implented
            x1 (float): Coordinate of right boundary. Note that only +1 is currently implented
        """
        # need linear transformation y = ax + b with a = (x1-x0)/2 and b = (x1+x0)/2
        self.lin_trf_fac = (x1 - x0) / 2
        self.lin_trf_off = (x1 + x0) / 2
        super().__init__(*args, x0=x0, x1=x1, **kwargs)

        self.norm = self.get_norm()

    def get_grid(self):
        """
        Generates a 1D grid with Chebychev points. These are clustered at the boundary. You need this kind of grid to
        use discrete cosine transformation (DCT) to get the Chebychev representation. If you want a different grid, you
        need to do an affine transformation before any Chebychev business.

        Returns:
            numpy.ndarray: 1D grid
        """
        return (
            self.lin_trf_fac
            * self.xp.cos(np.pi / self.N * (self.xp.arange(self.N) + 0.5))
            + self.lin_trf_off
        )

    def get_wavenumbers(self):
        """Get the domain in spectral space"""
        return self.xp.arange(self.N)

    @cache
    def get_conv(self, name, N=None):
        """
        Get conversion matrix between different kinds of polynomials. The supported kinds are
         - T: Chebychev polynomials of first kind
         - U: Chebychev polynomials of second kind
         - D: Dirichlet recombination.

        You get the desired matrix by choosing a name as ``A2B``. I.e. ``T2U`` for the conversion matrix from T to U.
        Once generates matrices are cached. So feel free to call the method as often as you like.

        Args:
         name (str): Conversion code, e.g. 'T2U'
         N (int): Size of the matrix (optional)

        Returns:
            scipy.sparse: Sparse conversion matrix
        """
        N = N if N else self.N
        sp = self.sparse_lib

        def get_forward_conv(name):
            if name == "T2U":
                mat = (sp.eye(N) - sp.eye(N, k=2)).tocsc() / 2.0
                mat[:, 0] *= 2
            elif name == "D2T":
                mat = sp.eye(N) - sp.eye(N, k=2)
            elif name[0] == name[-1]:
                mat = self.sparse_lib.eye(self.N)
            else:
                raise NotImplementedError(f"Don't have conversion matrix {name!r}")
            return mat

        try:
            mat = get_forward_conv(name)
        except NotImplementedError as E:
            try:
                fwd = get_forward_conv(name[::-1])
                import scipy.sparse as sp

                if self.sparse_lib == sp:
                    mat = self.sparse_lib.linalg.inv(fwd.tocsc())
                else:
                    mat = self.sparse_lib.csc_matrix(sp.linalg.inv(fwd.tocsc().get()))
            except NotImplementedError:
                raise NotImplementedError from E

        return mat

    def get_basis_change_matrix(self, conv="T2T", **kwargs):
        """
        As the differentiation matrix in Chebychev-T base is dense but is sparse when simultaneously changing base to
        Chebychev-U, you may need a basis change matrix to transfer the other matrices as well. This function returns a
        conversion matrix from `ChebychevHelper.get_conv`. Not that `**kwargs` are used to absorb arguments for other
        bases, see documentation of `SpectralOneDBase.get_basis_change_matrix`.

        Args:
            conv (str): Conversion code, i.e. T2U

        Returns:
            Sparse conversion matrix
        """
        return self.get_conv(conv)

    def get_integration_matrix(self, lbnd=0):
        """
        Get matrix for integration

        Args:
            lbnd (float): Lower bound for integration, only 0 is currently implemented

        Returns:
           Sparse integration matrix
        """
        S = self.sparse_lib.diags(
            1 / (self.xp.arange(self.N - 1) + 1), offsets=-1
        ) @ self.get_conv("T2U")
        n = self.xp.arange(self.N)
        if lbnd == 0:
            S = S.tocsc()
            S[0, 1::2] = (
                (n / (2 * (self.xp.arange(self.N) + 1)))[1::2]
                * (-1) ** (self.xp.arange(self.N // 2))
                / (np.append([1], self.xp.arange(self.N // 2 - 1) + 1))
            ) * self.lin_trf_fac
        else:
            raise NotImplementedError(
                f"This function allows to integrate only from x=0, you attempted from x={lbnd}."
            )
        return S

    def get_integration_weights(self):
        """Weights for integration across entire domain"""
        n = self.xp.arange(self.N, dtype=float)

        weights = (-1) ** n + 1
        weights[2:] /= 1 - (n**2)[2:]

        weights /= 2 / self.L
        return weights

    def get_differentiation_matrix(self, p=1):
        """
        Keep in mind that the T2T differentiation matrix is dense.

        Args:
            p (int): Derivative you want to compute

        Returns:
            numpy.ndarray: Differentiation matrix
        """
        D = self.xp.zeros((self.N, self.N))
        for j in range(self.N):
            for k in range(j):
                D[k, j] = 2 * j * ((j - k) % 2)

        D[0, :] /= 2
        return (
            self.sparse_lib.csc_matrix(self.xp.linalg.matrix_power(D, p))
            / self.lin_trf_fac**p
        )

    @cache
    def get_norm(self, N=None):
        """
        Get normalization for converting Chebychev coefficients and DCT

        Args:
            N (int, optional): Resolution

        Returns:
            self.xp.array: Normalization
        """
        N = self.N if N is None else N
        norm = self.xp.ones(N) / N
        norm[0] /= 2
        return norm

    def transform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        DCT along axes. `kwargs` will be passed on to the FFT library.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform along

        Returns:
            Data in spectral space
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs["s"] = shape
        kwargs["norm"] = kwargs.get("norm", "backward")

        trf = self.fft_lib.dctn(u, *args, axes=axes, type=2, **kwargs)
        for axis in axes:
            if self.N < trf.shape[axis]:
                # mpi4py-fft implements padding only for FFT, where the frequencies are sorted such that the zeros are
                # removed in the middle rather than the end. We need to resort this here and put the highest frequencies
                # in the middle.
                _trf = self.xp.zeros_like(trf)
                N = self.N
                N_pad = _trf.shape[axis] - N
                end_first_half = N // 2 + 1

                # copy first "half"
                su = [slice(None)] * trf.ndim
                su[axis] = slice(0, end_first_half)
                _trf[tuple(su)] = trf[tuple(su)]

                # copy second "half"
                su = [slice(None)] * u.ndim
                su[axis] = slice(end_first_half + N_pad, None)
                s_u = [slice(None)] * u.ndim
                s_u[axis] = slice(end_first_half, N)
                _trf[tuple(su)] = trf[tuple(s_u)]

                trf = _trf

            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)
            norm = self.xp.ones(trf.shape[axis]) * self.norm[-1]
            norm[: self.N] = self.norm
            trf *= norm[(*expansion,)]
        return trf

    def itransform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        Inverse DCT along axis.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform along

        Returns:
            Data in physical space
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs["s"] = shape
        kwargs["norm"] = kwargs.get("norm", "backward")
        kwargs["overwrite_x"] = kwargs.get("overwrite_x", False)

        for axis in axes:
            if self.N == u.shape[axis]:
                _u = u.copy()
            else:
                # mpi4py-fft implements padding only for FFT, where the frequencies are sorted such that the zeros are
                # added in the middle rather than the end. We need to resort this here and put the padding in the end.
                N = self.N
                _u = self.xp.zeros_like(u)

                # copy first half
                su = [slice(None)] * u.ndim
                su[axis] = slice(0, N // 2 + 1)
                _u[tuple(su)] = u[tuple(su)]

                # copy second half
                su = [slice(None)] * u.ndim
                su[axis] = slice(-(N // 2), None)
                s_u = [slice(None)] * u.ndim
                s_u[axis] = slice(N // 2, N // 2 + (N // 2))
                _u[tuple(s_u)] = u[tuple(su)]

                if N % 2 == 0:
                    su = [slice(None)] * u.ndim
                    su[axis] = N // 2
                    _u[tuple(su)] *= 2

            # generate norm
            expansion = [np.newaxis for _ in u.shape]
            expansion[axis] = slice(0, u.shape[axis], 1)
            norm = self.xp.ones(_u.shape[axis])
            norm[: self.N] = self.norm
            norm = self.get_norm(u.shape[axis]) * _u.shape[axis] / self.N

            _u /= norm[(*expansion,)]

        return self.fft_lib.idctn(_u, *args, axes=axes, type=2, **kwargs)

    def get_BC(self, kind, **kwargs):
        """
        Get boundary condition row for boundary bordering. `kwargs` will be passed on to implementations of the BC of
        the kind you choose. Specifically, `x` for `'dirichlet'` boundary condition, which is the coordinate at which to
        set the BC.

        Args:
            kind ('integral' or 'dirichlet'): Kind of boundary condition you want
        """
        if kind.lower() == "integral":
            return self.get_integ_BC_row(**kwargs)
        elif kind.lower() == "dirichlet":
            return self.get_Dirichlet_BC_row(**kwargs)
        elif kind.lower() == "neumann":
            return self.get_Neumann_BC_row(**kwargs)
        else:
            return super().get_BC(kind)

    def get_integ_BC_row(self):
        """
        Get a row for generating integral BCs with T polynomials.
        It returns the values of the integrals of T polynomials over the entire interval.

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        n = self.xp.arange(self.N) + 1
        me = self.xp.zeros_like(n).astype(float)
        me[2:] = ((-1) ** n[1:-1] + 1) / (1 - n[1:-1] ** 2)
        me[0] = 2.0
        return me

    def get_Dirichlet_BC_row(self, x):
        """
        Get a row for generating Dirichlet BCs at x with T polynomials.
        It returns the values of the T polynomials at x.

        Args:
            x (float): Position of the boundary condition

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        if x == -1:
            return (-1) ** self.xp.arange(self.N)
        elif x == 1:
            return self.xp.ones(self.N)
        elif x == 0:
            n = (1 + (-1) ** self.xp.arange(self.N)) / 2
            n[2::4] *= -1
            return n
        else:
            raise NotImplementedError(
                f"Don't know how to generate Dirichlet BC's at {x=}!"
            )

    def get_Neumann_BC_row(self, x):
        """
        Get a row for generating Neumann BCs at x with T polynomials.

        Args:
            x (float): Position of the boundary condition

        Returns:
            self.xp.ndarray: Row to put into a matrix
        """
        n = self.xp.arange(self.N, dtype="D")
        nn = n**2
        if x == -1:
            me = nn
            me[1:] *= (-1) ** n[:-1]
            return me
        elif x == 1:
            return nn
        else:
            raise NotImplementedError(
                f"Don't know how to generate Neumann BC's at {x=}!"
            )

    def get_Dirichlet_recombination_matrix(self):
        """
        Get matrix for Dirichlet recombination, which changes the basis to have sparse boundary conditions.
        This makes for a good right preconditioner.

        Returns:
            scipy.sparse: Sparse conversion matrix
        """
        N = self.N
        sp = self.sparse_lib

        return sp.eye(N) - sp.eye(N, k=2)
