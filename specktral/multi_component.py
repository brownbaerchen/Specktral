import numpy as np

from .utils import eliminate_zeros


class MultiComponent:
    xp = np

    def __init__(self, base, components):
        self.base = base
        self.components = components

        self.sparse_lib = base.sparse_lib

    @property
    def ncomponents(self):
        return len(self.components)

    def empty_physical(self):
        shape = [
            self.ncomponents,
        ] + [self.base.N]
        return self.xp.empty(shape)

    def index(self, name):
        """
        Get the index of component `name`.

        Args:
            name (str or list of strings): Name(s) of component(s)

        Returns:
            int: Index of the component
        """
        if type(name) in [str, int]:
            return self.components.index(name)
        elif type(name) in [list, tuple]:
            return (self.index(me) for me in name)
        else:
            raise NotImplementedError(
                f"Don't know how to compute index for {type(name)=}"
            )

    def transform(self, u, axes=None):
        axes = axes if axes else tuple(i for i in range(self.base.ndim))
        axes = tuple([axis + 1 for axis in axes])

        return self.base.transform(u, axes=axes)

    def itransform(self, u_hat, axes=None):
        axes = axes if axes else tuple(i for i in range(self.base.ndim))
        axes = tuple([axis + 1 for axis in axes])

        return self.base.itransform(u_hat, axes=axes)

    def get_empty_operator_matrix(self, diag=False):
        """
        Return a matrix of operators to be filled with the connections between the solution components.

        Args:
            diag (bool): Whether operator is block-diagonal

        Returns:
            list containing sparse zeros
        """
        S = len(self.components)
        Zero = self.base.get_Id() * 0
        if diag:
            return [Zero for _ in range(S)]
        else:
            return [[Zero for _ in range(S)] for _ in range(S)]

    def add_equation_to_operator(self, A, equation, relations):
        """
        Add the left hand part (that you want to solve implicitly) of an equation to a list of lists of sparse matrices
        that you will convert to an operator later.

        Example:
        Setup linear operator `L` for 1D heat equation using Chebychev method in first order form and T-to-U
        preconditioning:

        .. code-block:: python
            helper = CompositeBase()

            helper.add_axis(base='chebychev', N=8)
            helper.add_component(['u', 'ux'])
            helper.setup_fft()

            I = helper.get_Id()
            Dx = helper.get_differentiation_matrix(axes=(0,))
            T2U = helper.get_basis_change_matrix('T2U')

            L_lhs = {
                'ux': {'u': -T2U @ Dx, 'ux': T2U @ I},
                'u': {'ux': -(T2U @ Dx)},
            }

            operator = helper.get_empty_operator_matrix()
            for line, equation in L_lhs.items():
                helper.add_equation_lhs(operator, line, equation)

            L = helper.convert_operator_matrix_to_operator(operator)

        Args:
            A (list of lists of sparse matrices): The operator to be
            equation (str): The equation of the component you want this in
            relations: (dict): Relations between quantities
        """
        for k, v in relations.items():
            A[self.index(equation)][self.index(k)] = v

    def convert_operator_matrix_to_operator(self, M):
        """
        Promote the list of lists of sparse matrices to a single sparse matrix that can be used as linear operator.
        See documentation of `CompositeBase.add_equation_lhs` for an example.

        Args:
            M (list of lists of sparse matrices): The operator to be

        Returns:
            sparse linear operator
        """
        if len(self.components) == 1:
            op = M[0][0]
        else:
            op = self.sparse_lib.bmat(M, format="csc")

        op = eliminate_zeros(self.sparse_lib, op)
        return op

    def setup_operator(self, LHS):
        """
        Setup a sparse linear operator by adding relationships. See documentation for ``GenericSpectralLinear.setup_L`` to learn more.

        Args:
            LHS (dict): Equations to be added to the operator

        Returns:
            sparse linear operator
        """
        operator = self.get_empty_operator_matrix()
        for line, equation in LHS.items():
            self.add_equation_to_operator(operator, line, equation)
        return self.convert_operator_matrix_to_operator(operator)
