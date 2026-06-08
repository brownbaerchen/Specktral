import pytest


def test_NotImplementedErrors():
    from specktral.bases.core import SpectralOneDBase

    base = SpectralOneDBase(N=1, x0=0, x1=1)

    funcs = [
        "get_grid",
        "get_differentiation_matrix",
        "get_integration_matrix",
        "get_wavenumbers",
    ]
    for func in funcs:
        with pytest.raises(NotImplementedError):
            getattr(base, func)()

    with pytest.raises(NotImplementedError, match="weird BC"):
        base.get_BC("weird BC")
