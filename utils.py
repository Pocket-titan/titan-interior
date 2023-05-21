# %%
from textwrap import dedent
from typing import Union

import numpy as np
import seafreeze.seafreeze as sf
from burnman import Mineral, minerals
from burnman.tools.chemistry import dictionarize_formula, formula_mass


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


def integrate_density_and_temp(layers, values, T0: Union[None, float] = 93):
    num_steps = values.shape[1]
    values = np.copy(values)

    # Downwards
    for i, layer in reversed([*enumerate(layers)]):
        [r0, r1, rho_0, alpha, K, Cp] = layer[:6]
        dr = (r1 - r0) / num_steps

        is_innermost_layer = i == 0
        is_outermost_layer = i == len(layers) - 1
        prev_layer_idx = i + 1
        first_step = -1
        last_step = 0

        if T0 is None:
            Ts = values[i, :, 5]
        else:
            Ts = np.zeros(num_steps)
            Ts[0] = T = T0 if is_outermost_layer else values[prev_layer_idx, last_step, 5]

            # Propagate T
            for j in range(num_steps):
                g = values[i, num_steps - j - 1, 2]

                _Cp = Cp(T) if callable(Cp) else Cp
                _alpha = alpha(T) if callable(alpha) else alpha

                T += (_alpha * g * T / _Cp * dr) if _Cp != 0 else 0.0
                Ts[num_steps - j - 1] = T

            values[i, :, 5] = Ts

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
            print(rho - rho_0)
            rhos[num_steps - j - 1] = rho

        values[i, :, 4] = rhos

    return values


def iterate_layers(layers, ref_temps=None, num_steps=10_000, max_iterations=100) -> np.ndarray:
    converged = False
    iterations = 0
    values = None

    while not converged:
        new_values = integrate(layers, num_steps=10_000, values=values)

        if iterations == 0 and ref_temps is not None:
            new_values[:, :, 5] = ref_temps

        new_values = integrate_density_and_temp(
            layers,
            new_values,
            T0=93.6 if ref_temps is None else None,
        )

        if values is not None:
            diff = new_values - values
            print(diff.sum())
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


def adjust_mineral_density(mineral, rho_0):
    rho, V = mineral.evaluate(["rho", "V"], 10**5, 300)
    mass = rho * V
    V_new = mass / rho_0
    delta_V = V_new - V

    return Mineral(
        params={**mineral.params, "name": f"{mineral.params['name']} rho_0={rho_0:.3e}"},
        property_modifiers=[
            [
                "linear",
                {
                    "delta_S": 0,
                    "delta_E": 0,
                    "delta_V": delta_V,
                },
            ]
        ],
    )


def get_ice_values(phase: str, rho_0=None):
    assert phase in ["Ih", "II", "III", "V", "VI", "water1", "water2", "water_IAPWS95"]
    water_formula = dictionarize_formula("H2O")

    ref_p = 10**5  # Pa
    ref_T = 300  # K
    out = sf.seafreeze(
        np.array([ref_p / 10**6, ref_T]),  # MPa, K
        phase=phase,
    )

    molar_mass = formula_mass(water_formula)

    [Kt, Kp, Cv, Cp, alpha, rho, shear, H, S, V] = [
        out.Kt.item() * 1e6,
        out.Kp.item(),
        out.Cv.item(),
        out.Cp.item(),
        out.alpha.item(),
        out.rho.item(),
        out.shear.item() * 1e6,
        out.H.item(),
        out.S.item(),
        out.V.item() * molar_mass,
    ]

    params = {
        # Take liquid water as a base
        **minerals.HP_2011_ds62.h2oL().params,
        "name": f"Ice {phase}",
        "molar_mass": molar_mass,
        "n": sum(water_formula.values()),
        # Seafreeze values
        "K_0": Kt,
        "Kprime_0": Kp,
        "G_0": shear,
        "V_0": V,
        "H_0": H,
        "S_0": S,
        "P_0": ref_p,
        "T_0": ref_T,
        "grueneisen_0": alpha * Kt / (Cv * rho),
        "Cp": [Cp, 0, 0, 0],
        "Cv": [Cv, 0, 0, 0],
        "q_0": 1,
    }

    mineral = Mineral(params=params)

    if rho_0 is not None:
        mineral = adjust_mineral_density(mineral, rho_0)

    return (mineral, alpha, Kt, Cp)
