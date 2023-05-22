# %%
from textwrap import dedent
from typing import Union

import numpy as np


# %%
def create_layers(layers):
    """
    Pass layers in format:
    ```
    [
        [r0, r1, rho_0],
        ...
    ]
    or
    [
        [radius, rho],
        ...
    ]
    or
    [
        [r0, r1, rho_0, K, \\alpha],
        ...
    ]
    or
    [
        [radius, rho, K, \\alpha],
        ...
    ]
    or
        [radius,rho,K,T, \\alpha, cp or dT/dr, conductive or convective] (0 for conductive, 1 for convective)
    ```
    where K = bulk modulus, \\alpha = thermal expansivity
    """
    layers = np.array(layers)

    if layers.shape[1] in [2, 5]:
        new_layers = np.zeros([len(layers), layers.shape[1] + 1], dtype=object)

        for i in range(len(layers)):
            start = 0 if i == 0 else new_layers[i - 1, 1]
            end = start + layers[i, 0]
            assert (start + end) > 0
            new_layers[i, :] = [start, end, *layers[i, 1:]]

        return new_layers

    if layers.shape[1] in [3, 6]:
        for i, layer in enumerate(layers):
            [r0, r1, rho, *rest] = layer
            assert r0 >= 0 and r1 >= 0
            assert r1 > r0

            if i == 0:
                assert r0 == 0

            if i > 0:
                assert r0 == layers[i - 1][1]

        return layers

    if layers.shape[1] in [6, 7]:
        new_layers = np.zeros([len(layers), layers.shape[1] + 1])
        return new_layers

    raise Exception()


def compute_mass(values):
    values = values.reshape([values.shape[0] * values.shape[1], values.shape[2]])[::-1, :]
    m = 0

    for i in range(1, values.shape[0]):
        r0 = values[i, 0]
        r1 = values[i - 1, 0]
        rho = values[i, 4]
        m += 4 * np.pi * rho * (r1**3 - r0**3) / 3

    return m


def compute_moment_of_inertia(values):
    values = values.reshape([values.shape[0] * values.shape[1], values.shape[2]])[::-1, :]
    moment_of_inertia = 0

    for i in range(1, values.shape[0]):
        r0 = values[i, 0]
        r1 = values[i - 1, 0]
        rho = values[i, 4]
        moment_of_inertia += 8 * np.pi * rho * (r1**5 - r0**5) / 15

    return moment_of_inertia


def integrate_upwards(r0, dr, M0, rhos, num_steps=10_000):
    """
    Integrate a layer upwards and return [[r, M, g]]
    """
    r = r0
    M = M0
    G = 6.6743e-11

    interior = np.zeros([num_steps, 3])

    for i in range(num_steps):
        rho = rhos[i]
        r += dr
        M += 4 * np.pi * rho * (r**2) * dr
        g = G * M / (r**2)

        interior[i, 0] = r
        interior[i, 1] = M
        interior[i, 2] = g

    return interior


def integrate_downwards(p0, dr, rhos, interior):
    """
    Integrate a layer downwards and return an array with the pressures
    """
    num_steps = len(interior)
    ps = np.zeros(num_steps)
    ps[0] = p = p0

    for i in range(num_steps):
        rho = interior[num_steps - i - 1, 4]
        g = interior[num_steps - i - 1, 2]
        p += rho * g * dr
        ps[num_steps - i - 1] = p

    return ps


def integrate(layers, values=None, num_steps=10_000):
    if values is None:
        # [r, m, g, p, rho, T]
        values = np.zeros([len(layers), num_steps, 6])
        # Assume constant density on first run
        for i, layer in enumerate(layers):
            rho_0 = layer[2]
            values[i, :, 4] = rho_0
    else:
        assert values.shape[0] == len(layers) and values.shape[2] == 6

    # Upwards
    for i, layer in enumerate(layers):
        [r0, r1, rho_0] = layer[:3]
        dr = (r1 - r0) / num_steps

        is_innermost_layer = i == 0
        is_outermost_layer = i == len(layers) - 1
        prev_layer_idx = i - 1
        first_step = 0
        last_step = -1

        rhos = values[i, :, 4]
        m0 = 0 if is_innermost_layer else values[prev_layer_idx, last_step, 1]
        interior = integrate_upwards(r0, dr, m0, rhos, num_steps)

        values[i, :, :3] = interior

    # Downwards
    for i, layer in reversed([*enumerate(layers)]):
        [r0, r1, rho_0] = layer[:3]
        dr = (r1 - r0) / num_steps

        is_innermost_layer = i == 0
        is_outermost_layer = i == len(layers) - 1
        prev_layer_idx = i + 1
        first_step = -1
        last_step = 0

        rhos = values[i, :, 4]
        p0 = 0 if is_outermost_layer else values[prev_layer_idx, last_step, 3]
        ps = integrate_downwards(p0, dr, rhos, values[i, :, :])

        values[i, :, 3] = ps

    return values


def integrate_density(layers, values):
    num_steps = values.shape[1]
    values = np.copy(values)

    # Downwards
    for i, layer in reversed([*enumerate(layers)]):
        [r0, r1, rho_0, alpha, K, Cp] = layer[:6]
        Ts = values[i, :, 5]

        rhos = np.zeros(num_steps)

        # Propagate rho
        for j in range(num_steps):
            T = Ts[num_steps - j - 1]
            dT = 0 if j == 0 else T - Ts[num_steps - j]

            p = values[i, num_steps - j - 1, 3]
            dp = 0 if j == 0 else p - values[i, num_steps - j, 3]

            _K = K(T) if callable(K) else K
            _alpha = alpha(T) if callable(alpha) else alpha

            rho = rho_0 * (1 - _alpha * dT + 1 / _K * dp)
            rhos[num_steps - j - 1] = rho

        values[i, :, 4] = rhos

    return values


def propagate_adiabatic(layer, values, T0, num_steps=10_000):
    [r0, r1, rho_0, alpha, K, Cp] = layer[:6]
    dr = (r1 - r0) / num_steps

    Ts = np.zeros(num_steps)
    Ts[0] = T = T0

    for j in range(num_steps):
        g = values[num_steps - j - 1, 2]

        _Cp = Cp(T) if callable(Cp) else Cp
        _alpha = alpha(T) if callable(alpha) else alpha

        T += (_alpha * g * T / _Cp * dr) if _Cp != 0 else 0.0
        Ts[num_steps - j - 1] = T

    return Ts[::-1]


def create_temps(layers, values, temp_info):
    assert len(temp_info) in [0, len(layers)]
    temps = []

    for i, layer in enumerate(reversed(layers)):
        j = len(layers) - i - 1

        T0 = temp_info[i]["top"] if "top" in temp_info[i] else temps[-1][-1]

        if "mode" in temp_info[i] and temp_info[i]["mode"] == "adiabatic":
            Ts = propagate_adiabatic(layer, values[j], T0)
        else:
            gradient = np.inf
            assert "gradient" in temp_info[i] or "bottom" in temp_info[i]
            if "gradient" in temp_info[i]:
                gradient = temp_info[i]["gradient"]
            if "bottom" in temp_info[i]:
                gradient = (temp_info[i]["bottom"] - T0) / (layer[1] / 1e3 - layer[0] / 1e3)

            Ts = np.linspace(
                T0,
                T0 + gradient * (layer[1] / 1e3 - layer[0] / 1e3),
                num=values.shape[1],
            )

        temps.append(Ts)

    return np.reshape([x[::-1] for x in temps][::-1], (values.shape[0], -1))


def iterate_layers(layers, temp_info=None, num_steps=10_000, max_iterations=100) -> np.ndarray:
    converged = False
    iterations = 0
    values = None

    if temp_info is None:
        temp_info = [
            {"top": 93, "mode": "adiabatic"},
            *([{"mode": "adiabatic"}] * (len(layers) - 1)),
        ]

    while not converged:
        new_values = integrate(layers, num_steps=10_000, values=values)

        if values is None:
            Ts = create_temps(layers, new_values, temp_info)
            new_values[:, :, 5] = Ts

        new_values = integrate_density(layers, new_values)

        if values is not None:
            diff = new_values - values
            if iterations % 10 == 0:
                print(f"Diff: {diff.sum():>.2e}")
            if np.isnan(diff.sum()):
                break
            converged = np.all(np.abs(diff) < 1e-10) and np.sum(diff**2) < 1e-12

        iterations += 1
        values = new_values

    print(
        f"Converged after {iterations=}" if converged else f"Failed to converge after {iterations=}"
    )

    return np.array(values)


def verify_results(m_total, g_surface, p_center, MoI_computed):
    G = 6.67408e-11

    # Titan values
    M = 1.3452e23
    R = 2575e3
    g = 1.35
    MoI = 0.352
    pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)

    print(
        dedent(
            f"""
        Computed values:
        Total mass:          {m_total:>.2e} kg
        Pressure at center:  {p_center/1e9:>.2f} GPa
        Gravity at surface:  {g_surface:>.2f} m/s^2
        Moment of inertia:   {MoI_computed:>.3f}

        Errors:
        Mass error:          {abs(m_total - M)/M * 100:.2f}%
        Pressure error:      {abs(p_center - pc_theoretical)/pc_theoretical * 100:.2f}%
        Gravity error:       {abs(g_surface - g)/g * 100:.2f}%
        MoI error:           {abs(MoI_computed - MoI)/MoI * 100:.2f}%
    """
        )
    )


layer_map = {
    "Fortes": {
        "Pure water ice": [
            "Rock",
            "Ice VI",
            "Ice V",
            "Ice II",
            "Ice I",
        ],
        "Light ocean": [
            "Rock",
            "Ice VI",
            "Ice V",
            "Water",
            "Ice I",
        ],
        "Dense ocean": [
            "Rock",
            "Ice VI",
            "Water",
            "Ice I",
        ],
        "Pure iron inner core": [
            "Iron",
            "Rock",
            "Ice VI",
            "Ice V",
            "Ice II",
            "Ice I",
        ],
    },
    "Grasset": {
        "Core": [
            "Iron",
            "Silicates",
            "HP ices",
            "Water",
            "Ice I",
        ],
        "No core": [
            "Silicates",
            "HP ices",
            "Water",
            "Ice I",
        ],
    },
    "Kronrod": {
        "15% Fe core": [
            "Fe-Si",
            "Rock ice mantle",
            "Water",
            "Ice I",
        ],
    },
}
