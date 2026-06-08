import numpy as np
import scipy

import logging


class SpectralOneDBase:
    """
    Abstract base class for 1D spectral discretizations. Defines a common interface with parameters and functions that
    all bases need to have.

    When implementing new bases, please take care to use the modules that are supplied as class attributes to enable
    the code for GPUs.

    Attributes:
        N (int): Resolution
        x0 (float): Coordinate of left boundary
        x1 (float): Coordinate of right boundary
        L (float): Length of the domain
        useGPU (bool): Whether to use GPUs

    """

    fft_lib = scipy.fft
    sparse_lib = scipy.sparse
    linalg = scipy.sparse.linalg
    xp = np
    distributable = False

    def __init__(self, N, x0=None, x1=None, useGPU=False, useFFTW=False):
        """
        Constructor

        Args:
            N (int): Resolution
            x0 (float): Coordinate of left boundary
            x1 (float): Coordinate of right boundary
            useGPU (bool): Whether to use GPUs
            useFFTW (bool): Whether to use FFTW for the transforms
        """
        self.N = N
        self.x0 = x0
        self.x1 = x1
        self.L = x1 - x0
        self.useGPU = useGPU
        self.plans = {}
        self.logger = logging.getLogger(name=type(self).__name__)

        if useGPU:
            self.setup_GPU()
            self.logger.debug("Set up for GPU")
        else:
            self.setup_CPU(useFFTW=useFFTW)

        if useGPU and useFFTW:
            raise ValueError("Please run either on GPUs or with FFTW, not both!")

    @classmethod
    def setup_GPU(cls):
        """switch to GPU modules"""
        import cupy as cp
        import cupyx.scipy.sparse as sparse_lib
        import cupyx.scipy.sparse.linalg as linalg
        import cupyx.scipy.fft as fft_lib

        cls.xp = cp
        cls.sparse_lib = sparse_lib
        cls.linalg = linalg
        cls.fft_lib = fft_lib

    @classmethod
    def setup_CPU(cls, useFFTW=False):
        """switch to CPU modules"""

        cls.xp = np
        cls.sparse_lib = scipy.sparse
        cls.linalg = scipy.sparse.linalg

        if useFFTW:
            from mpi4py_fft import fftw

            cls.fft_backend = "fftw"
            cls.fft_lib = fftw
        else:
            cls.fft_backend = "scipy"
            cls.fft_lib = scipy.fft

        cls.fft_comm_backend = "MPI"

    def get_Id(self):
        """
        Get identity matrix

        Returns:
            sparse diagonal identity matrix
        """
        return self.sparse_lib.eye(self.N)

    def get_zero(self):
        """
        Get a matrix with all zeros of the correct size.

        Returns:
            sparse matrix with zeros everywhere
        """
        return 0 * self.get_Id()

    def get_differentiation_matrix(self):
        raise NotImplementedError()

    def get_integration_matrix(self):
        raise NotImplementedError()

    def get_integration_weights(self):
        """Weights for integration across entire domain"""
        raise NotImplementedError()

    def get_wavenumbers(self):
        """
        Get the grid in spectral space
        """
        raise NotImplementedError

    def get_empty_operator_matrix(self, S, Zero):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Args:
            S (int): Number of components in the solution
            Zero (sparse matrix): Zero matrix used for initialization

        Returns:
            list of lists containing sparse zeros
        """
        return [[Zero for _ in range(S)] for _ in range(S)]

    def get_basis_change_matrix(self, *args, **kwargs):
        """
        Some spectral discretization change the basis during differentiation. This method can be used to transfer
        between the various bases.

        This method accepts arbitrary arguments that may not be used in order to provide an easy interface for multi-
        dimensional bases. For instance, you may combine an FFT discretization with an ultraspherical discretization.
        The FFT discretization will always be in the same base, but the ultraspherical discretization uses a different
        base for every derivative. You can then ask all bases for transfer matrices from one ultraspherical derivative
        base to the next. The FFT discretization will ignore this and return an identity while the ultraspherical
        discretization will return the desired matrix. After a Kronecker product, you get the 2D version of the matrix
        you want. This is what the `SpectralHelper` does when you call the method of the same name on it.

        Returns:
            sparse bases change matrix
        """
        return self.sparse_lib.eye(self.N)

    def get_BC(self, kind):
        """
        To facilitate boundary conditions (BCs) we use either a basis where all functions satisfy the BCs automatically,
        as is the case in FFT basis for periodic BCs, or boundary bordering. In boundary bordering, specific lines in
        the matrix are replaced by the boundary conditions as obtained by this method.

        Args:
            kind (str): The type of BC you want to implement please refer to the implementations of this method in the
            individual 1D bases for what is implemented

        Returns:
            self.xp.array: Boundary condition
        """
        raise NotImplementedError(f"No boundary conditions of {kind=!r} implemented!")

    def get_filter_matrix(self, kmin=0, kmax=None):
        """
        Get a bandpass filter.

        Args:
            kmin (int): Lower limit of the bandpass filter
            kmax (int): Upper limit of the bandpass filter

        Returns:
            sparse matrix
        """

        k = abs(self.get_wavenumbers())

        kmax = max(k) if kmax is None else kmax

        mask = self.xp.logical_or(k >= kmax, k < kmin)

        if self.useGPU:
            Id = self.get_Id().get()
        else:
            Id = self.get_Id()
        F = Id.tolil()
        F[:, mask] = 0
        return F.tocsc()

    def get_1dgrid(self):
        """
        Get the grid in physical space

        Returns:
            self.xp.array: Grid
        """
        raise NotImplementedError
