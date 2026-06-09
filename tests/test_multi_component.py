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

    u_hat = eq.transform(u)
    get_hat = (operator @ u_hat.flatten()).reshape(u_hat.shape)
    get = eq.itransform(get_hat)

    assert xp.allclose(expect, get)
