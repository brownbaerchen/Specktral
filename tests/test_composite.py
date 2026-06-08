import pytest
import numpy as np


def test_composite_instantiation():
    from specktral.composite import CompositeBase
    from specktral.bases import Fourier, Chebychev

    x_base = Fourier(N=128)
    y_base = Chebychev(N=64)

    with pytest.raises(AssertionError):
        _ = CompositeBase(x_base)

    _ = CompositeBase([x_base, y_base])


def test_composite_grid():
    from specktral.composite import CompositeBase
    from specktral.bases import Fourier, Chebychev

    x_base = Fourier(N=128)
    y_base = Chebychev(N=64)
    z_base = Fourier(N=17)

    # 2D
    composite = CompositeBase([x_base, y_base])
    grid = composite.get_grid()
    assert len(grid) == 2
    assert np.allclose([me.shape for me in grid], (x_base.N, y_base.N))
    for i in range(y_base.N):
        assert np.allclose(grid[0][:, i], x_base.get_grid())
    for i in range(x_base.N):
        assert np.allclose(grid[1][i, :], y_base.get_grid())

    # 3D
    composite = CompositeBase([x_base, y_base, z_base])
    grid = composite.get_grid()
    assert len(grid) == 3
    assert np.allclose([me.shape for me in grid], (x_base.N, y_base.N, z_base.N))
    for i in range(y_base.N):
        for j in range(z_base.N):
            assert np.allclose(grid[0][:, i, j], x_base.get_grid())
    for i in range(x_base.N):
        for j in range(z_base.N):
            assert np.allclose(grid[1][i, :, j], y_base.get_grid())
    for i in range(x_base.N):
        for j in range(y_base.N):
            assert np.allclose(grid[2][i, j, :], z_base.get_grid())
