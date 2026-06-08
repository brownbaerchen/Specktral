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


def test_composite_wave_number_grid():
    from specktral.composite import CompositeBase
    from specktral.bases import Fourier, Chebychev

    x_base = Fourier(N=128)
    y_base = Chebychev(N=64)
    z_base = Fourier(N=17)

    # 2D
    composite = CompositeBase([x_base, y_base])
    grid = composite.get_wavenumbers()
    assert len(grid) == 2
    assert np.allclose([me.shape for me in grid], (x_base.N, y_base.N))
    for i in range(y_base.N):
        assert np.allclose(grid[0][:, i], x_base.get_wavenumbers())
    for i in range(x_base.N):
        assert np.allclose(grid[1][i, :], y_base.get_wavenumbers())

    # 3D
    composite = CompositeBase([x_base, y_base, z_base])
    grid = composite.get_wavenumbers()
    assert len(grid) == 3
    assert np.allclose([me.shape for me in grid], (x_base.N, y_base.N, z_base.N))
    for i in range(y_base.N):
        for j in range(z_base.N):
            assert np.allclose(grid[0][:, i, j], x_base.get_wavenumbers())
    for i in range(x_base.N):
        for j in range(z_base.N):
            assert np.allclose(grid[1][i, :, j], y_base.get_wavenumbers())
    for i in range(x_base.N):
        for j in range(y_base.N):
            assert np.allclose(grid[2][i, j, :], z_base.get_wavenumbers())


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_differentiation_matrix(dim):
    from specktral.composite import CompositeBase
    from specktral.bases import Fourier, Chebychev

    x_base = Fourier(N=128)
    y_base = Chebychev(N=64)
    z_base = Fourier(N=17)

    bases = [x_base, y_base, z_base][:dim]

    composite = CompositeBase(bases)
    grid = composite.get_grid()

    Dx = composite.get_differentiation_matrix(axes=(0,))
    u = np.sin(2 * grid[0])
    du_expect = 2 * np.cos(2 * grid[0])

    u_hat = composite.transform(u)
    du_hat = (Dx @ u_hat.flatten()).reshape(u_hat.shape)
    du = composite.itransform(du_hat)
    assert np.allclose(du, du_expect)

    if dim > 1:
        Dy = composite.get_differentiation_matrix(axes=(1,))
        u = 4 * grid[1] ** 2 + 3
        du_expect = 8 * grid[1]

        u_hat = composite.transform(u)
        du_hat = (Dy @ u_hat.flatten()).reshape(u_hat.shape)
        du = composite.itransform(du_hat)
        assert np.allclose(du, du_expect)

    if dim == 3:
        Dxz = composite.get_differentiation_matrix(axes=(0, 2))
        u = np.sin(2 * grid[0]) * np.cos(4 * grid[2])
        du_expect = 2 * np.cos(2 * grid[0]) * (-4) * np.sin(4 * grid[2])

        u_hat = composite.transform(u)
        du_hat = (Dxz @ u_hat.flatten()).reshape(u_hat.shape)
        du = composite.itransform(du_hat)
        assert np.allclose(du, du_expect)
