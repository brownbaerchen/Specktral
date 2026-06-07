import numpy as np
import scipy

from ..utils import cache
from .core import SpectralOneDBase

from functools import partial


class Fourier(SpectralOneDBase):
    distributable = True

    def __init__(self, *args, x0=0, x1=2 * np.pi, **kwargs):
        """
        Constructor.
        Please refer to the parent class for additional arguments. Notably, you have to supply a resolution `N` and you
        may choose to run on GPUs via the `useGPU` argument.

        Args:
            x0 (float, optional): Coordinate of left boundary
            x1 (float, optional): Coordinate of right boundary
        """
        super().__init__(*args, x0=x0, x1=x1, **kwargs)

    def get_1dgrid(self):
        """
        We use equally spaced points including the left boundary and not including the right one, which is the left boundary.
        """
        dx = self.L / self.N
        return self.xp.arange(self.N) * dx + self.x0

    def get_wavenumbers(self):
        """
        Be careful that this ordering is very unintuitive.
        """
        return self.xp.fft.fftfreq(self.N, 1.0 / self.N) * 2 * np.pi / self.L

    def get_differentiation_matrix(self, p=1):
        """
        This matrix is diagonal, allowing to invert concurrently.

        Args:
            p (int): Order of the derivative

        Returns:
            sparse differentiation matrix
        """
        k = self.get_wavenumbers()

        if self.useGPU:
            if p > 1:
                # Have to raise the matrix to power p on CPU because the GPU equivalent is not implemented in CuPy at the time of writing.
                from scipy.sparse.linalg import matrix_power

                D = self.sparse_lib.diags(1j * k).get()
                return self.sparse_lib.csc_matrix(matrix_power(D, p))
            else:
                return self.sparse_lib.diags(1j * k)
        else:
            return self.linalg.matrix_power(self.sparse_lib.diags(1j * k), p)

    def get_integration_matrix(self, p=1):
        """
        Get integration matrix to compute `p`-th integral over the entire domain.

        Args:
            p (int): Order of integral you want to compute

        Returns:
            sparse integration matrix
        """
        k = self.xp.array(self.get_wavenumbers(), dtype='complex128')
        k[0] = 1j * self.L
        return self.linalg.matrix_power(self.sparse_lib.diags(1 / (1j * k)), p)

    def get_integration_weights(self):
        """Weights for integration across entire domain"""
        weights = self.xp.zeros(self.N)
        weights[0] = self.L / self.N
        return weights

    def get_plan(self, u, forward, *args, **kwargs):
        if self.fft_lib.__name__ == 'mpi4py_fft.fftw':
            if 'axes' in kwargs.keys():
                kwargs['axes'] = tuple(kwargs['axes'])
            key = (forward, u.shape, args, *(me for me in kwargs.values()))
            if key in self.plans.keys():
                return self.plans[key]
            else:
                self.logger.debug(f'Generating FFT plan for {key=}')
                transform = self.fft_lib.fftn(u, *args, **kwargs) if forward else self.fft_lib.ifftn(u, *args, **kwargs)
                self.plans[key] = transform

            return self.plans[key]
        else:
            if forward:
                return partial(self.fft_lib.fftn, norm=kwargs.get('norm', 'backward'))
            else:
                return partial(self.fft_lib.ifftn, norm=kwargs.get('norm', 'forward'))

    def transform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        FFT along axes. `kwargs` are passed on to the FFT library.

        Args:
            u: Data you want to transform
            axes (tuple): Axes you want to transform over

        Returns:
            transformed data
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        plan = self.get_plan(u, *args, forward=True, axes=axes, **kwargs)
        return plan(u, *args, axes=axes, **kwargs)

    def itransform(self, u, *args, axes=None, shape=None, **kwargs):
        """
        Inverse FFT.

        Args:
            u: Data you want to transform
            axes (tuple): Axes over which to transform

        Returns:
            transformed data
        """
        axes = axes if axes else tuple(i for i in range(u.ndim))
        kwargs['s'] = shape
        plan = self.get_plan(u, *args, forward=False, axes=axes, **kwargs)
        return plan(u, *args, axes=axes, **kwargs) / np.prod([u.shape[axis] for axis in axes])

    def get_BC(self, kind):
        """
        Get a sort of boundary condition. You can use `kind=integral`, to fix the integral, or you can use `kind=Nyquist`.
        The latter is not really a boundary condition, but is used to set the Nyquist mode to some value, preferably zero.
        You should set the Nyquist mode zero when the solution in physical space is real and the resolution is even.

        Args:
            kind ('integral' or 'nyquist'): Kind of BC

        Returns:
            self.xp.ndarray: Boundary condition row
        """
        if kind.lower() == 'integral':
            return self.get_integ_BC_row()
        elif kind.lower() == 'nyquist':
            assert (
                self.N % 2 == 0
            ), f'Do not eliminate the Nyquist mode with odd resolution as it is fully resolved. You chose {self.N} in this axis'
            BC = self.xp.zeros(self.N)
            BC[self.get_Nyquist_mode_index()] = 1
            return BC
        else:
            return super().get_BC(kind)

    def get_Nyquist_mode_index(self):
        """
        Compute the index of the Nyquist mode, i.e. the mode with the lowest wavenumber, which doesn't have a positive
        counterpart for even resolution. This means real waves of this wave number cannot be properly resolved and you
        are best advised to set this mode zero if representing real functions on even-resolution grids is what you're
        after.

        Returns:
            int: Index of the Nyquist mode
        """
        k = self.get_wavenumbers()
        Nyquist_mode = min(k)
        return self.xp.where(k == Nyquist_mode)[0][0]

    def get_integ_BC_row(self):
        """
        Only the 0-mode has non-zero integral with FFT basis in periodic BCs
        """
        me = self.xp.zeros(self.N)
        me[0] = self.L / self.N
        return me
