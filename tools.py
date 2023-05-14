import numpy as np


def integrate_layers(layers, num_steps=1000):
    """
    Layers start from center:
    [
        [0, 500, 1000],
        [500, 6371 * 1e3, 2000],
    ]

    Returns:
    [layer, step, [r, m, g, p]]
    """
    assert layers[0][0] == 0
    values = np.zeros([len(layers), num_steps, 4])

    for i, layer in enumerate(layers):
        [r0, r1, rho_0] = layer
        assert r1 > r0

        if i > 0:
            if not r0 == layers[i - 1][1]:
                raise Exception("Layers must be contiguous")

        if i < len(layers) - 1:
            if not r1 == layers[i + 1][0]:
                raise Exception("Layers must be contiguous")

        dr = (r1 - r0) / num_steps
        # if we are starting at the center, take 0, else take the last mass of the last layer
        M = 0 if i == 0 else values[i - 1, -1, 1]
        interior = integrate_upwards(r0, dr, M, rho_0, num_steps)

        values[i, :, :] = interior

    for j, layer in enumerate(reversed(layers)):
        i = len(layers) - j - 1
        [r0, r1, rho_0] = layer

        assert r1 > r0
        dr = (r1 - r0) / num_steps
        # if we are starting at the surface, take 0, else take the last pressure of the last layer
        p0 = 0 if j == 0 else values[i + 1, 0, 3]
        ps = integrate_downwards(p0, dr, rho_0, values[i, :, :])

        values[i, :, 3] = ps

    return values


def integrate_upwards(r0, dr, M0, rho_0, num_steps=1000):
    """
    Integrate a layer upwards and return [[r, M, g]]
    """
    r = r0
    M = M0
    G = 6.6743e-11

    interior = np.zeros([num_steps, 4])

    for i in range(num_steps):
        r += dr
        M += 4 * np.pi * rho_0 * (r**2) * dr
        g = G * M / (r**2)

        interior[i, 0] = r
        interior[i, 1] = M
        interior[i, 2] = g

    return interior


def integrate_downwards(p0, dr, rho_0, interior):
    """
    Integrate a layer downwards and return an array with the pressures
    """
    num_steps = len(interior)
    ps = np.zeros(num_steps)
    ps[0] = p = p0

    for i in range(num_steps):
        g = interior[num_steps - i - 1, 2]
        p += rho_0 * g * dr
        ps[num_steps - i - 1] = p

    return ps


def compute_mass(layers):
    """
    Add up the mass of layers as spherical shells with constant density
    """
    M = 0

    for layer in layers:
        [r0, r1, rho_0] = layer
        M += 4 * np.pi * rho_0 * (r1**3 - r0**3) / 3

    return M
