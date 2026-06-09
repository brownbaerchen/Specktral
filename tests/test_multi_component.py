def test_operator():
    from specktral.bases import Fourier
    from specktral.multi_component import MultiComponent

    base = Fourier(N=4)
    components = ["u", "v"]

    D = base.get_differentiation_matrix()
    Id = base.get_Id()

    eqs_LHS = {
        "u": {"v": D},
        "v": {
            "u": Id,
            "v": Id,
        },
    }

    eq = MultiComponent(base, components)
    operator = eq.setup_operator(eqs_LHS)

    x = base.get_grid()
    xp = eq.xp

    u = eq.empty_physical()
    u[0] = xp.sin(x)
    u[1] = xp.cos(x)

    expect = eq.empty_physical()
    expect[0] = -xp.sin(x)
    expect[1] = xp.sin(x) + xp.cos(x)

    u_hat = base.transform(u)
    get_hat = (operator @ u_hat.flatten()).reshape(u_hat.shape)
    get = base.itransform(get_hat)

    assert xp.allclose(expect, get)


def test_operator_2d():
    from specktral.bases import Fourier, Chebychev
    from specktral.composite import CompositeBase
    from specktral.multi_component import MultiComponent

    bases1D = [Fourier(N=4), Chebychev(N=3)]
    base = CompositeBase(bases1D)
    components = ["u", "v"]

    eq = MultiComponent(base, components)

    D = base.get_differentiation_matrix(axes=(0,))
    Id = base.get_Id()

    eqs_LHS = {
        "u": {"v": D},
        "v": {
            "u": Id,
            "v": Id,
        },
    }

    eq = MultiComponent(base, components)
    operator = eq.setup_operator(eqs_LHS)

    X, Y = base.get_grid()
    xp = eq.xp

    u = eq.empty_physical()
    u[0] = xp.sin(X)
    u[1] = xp.cos(X)

    expect = eq.empty_physical()
    expect[0] = -xp.sin(X)
    expect[1] = xp.sin(X) + xp.cos(X)

    u_hat = base.transform(u)
    get_hat = (operator @ u_hat.flatten()).reshape(u_hat.shape)
    get = base.itransform(get_hat)

    assert xp.allclose(expect, get)
