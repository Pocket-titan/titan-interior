# %%
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tools import create_layers

np.set_printoptions(precision=2, suppress=True)

sns.set_theme(
    style="ticks",
    palette="Set2",
    rc={
        "axes.grid": True,
        "grid.linestyle": "--",
        "axes.spines.right": False,
        "axes.spines.top": False,
    },
)
pal = sns.color_palette("Set2")

G = 6.6743e-11

# Titan values
M = 1.3452e23
R = 2575e3
g = 1.35
MoI = 0.352


# %%
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
        # assume constant density on first run
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
            rhos[num_steps - j - 1] = rho

        values[i, :, 4] = rhos

    return values


def K(K0, C, D):
    return lambda T: K0 + C * T + D * T**2


# [rho_0, alpha, K, Cp]
thermo_values = {
    "Ice I": [
        934.31,  # Fortes (2012)
        1.56e-4,  # Grasset
        K(10.995e9, -0.004068, -2.051e-5),  # Fortes (2012)
        1925,  # Kronrod
    ],
    "Ice II": [
        1199.65,  # Fortes (2012)
        2.48e-4,  # Fortes (The incompressibility and thermal expansivity of D2O ice II...)
        K(13.951e9, 0.0014, -4.25e-5),  # Fortes (2012)
        1.85e3,  # Vladimir Tchijov, Heat capacity of high-pressure ice polymorphs (eyeballed avg)
    ],
    "Ice V": [
        1267.0,  # Fortes (2012)
        4.5e-5,  # Fortes (2012)
        13.93e9,  # Fortes (2012)
        2.5e3,  # Vladimir Tchijov, Heat capacity of high-pressure ice polymorphs (eyeballed avg)
    ],
    "Ice VI": [
        1326.8,  # Fortes (2012)
        211e-6,  # Noya (2007) Equation of State...
        K(17.82e9, -0.0385, 3.625e-5),  # Fortes (2012)
        2.3e3,  # Vladimir Tchijov, Heat capacity of high-pressure ice polymorphs (eyeballed avg)
    ],
    "Water": [
        1000,
        210e-6,
        2.2e9,
        4180,
    ],
    # "Rock" from Fortes (2012)
    "Rock": [
        (3343 + 2558) / 2,  # 50/50 of Antigorite and Olivine, Fortes
        (2.66 + 2) / 2 * 1e-5,  # Fortes
        lambda T: (67.27e9 - 0.01 * T + 131.1e9 - 0.0223 * T) / 2,  # Fortes
        lambda T: (1e3 + (289.73 - 0.024015 * T + 131045 / T**2 - 2779 / T ** (1 / 2)))
        / 2,
        # 50/50 average of values for Antigorite (Osako (2010) Thermal diffusivity, thermal
        # conductivity...) and Olivine (Robie, Heat capacity and entropy of Ni2sio4-olivine from..),
    ],
    "Iron": [
        8000,
        2e-5,  # Ichikawa (2014) (The P-V-T equation of state and thermodynamic properties of liquid iron)
        # K(109.7e9, 4.66, -0.043),
        # K(12.5e9, -0.0104, 0), # Fortes (2012)
        46e9,  # at 5 GPa (The P-V-T equation of state and thermodynamic properties of liquid iron)
        45,  # Ichikawa (2014)
    ],
    # Grasset
    "HP ices": [
        1310,  # Grasset
        4.5e-5,  #  Ice VI
        13.93e9,  # Ice VI
        2.5e3,  # Ice VI
    ],
    # Grasset
    "Silicates": [
        3300,  # Grasset
        2.4e-5,  # Grasset
        K(131.1e9, -0.0223, 0),  # Just assuming 100% Olivine from Fortes (2012)
        920,  # Grasset
    ],
    # Kronrod
    "Fe-Si": [
        3700,
        0,
        133e9,
        0,
    ],
    # Kronrod
    "Rock ice mantle": [
        2053.5,
        0,
        54.275e9,
        0,
    ],
    "Ice VII": [
        0,
        0,
        0,
        0,
    ],
}

models = {
    "Fortes": {
        "Pure water ice": create_layers(
            [
                [2054e3, *thermo_values["Rock"]],
                [132e3, *thermo_values["Ice VI"]],
                [117e3, *thermo_values["Ice V"]],
                [125e3, *thermo_values["Ice II"]],
                [146e3, *thermo_values["Ice I"]],
            ]
        ),
        "Light ocean": create_layers(
            [
                [2116e3, *thermo_values["Rock"]],
                [47e3, *thermo_values["Ice VI"]],
                [62e3, *thermo_values["Ice V"]],
                [250e3, 950, *thermo_values["Water"][1:]],
                [100e3, *thermo_values["Ice I"]],
            ]
        ),
        "Dense ocean": create_layers(
            [
                [1984e3, *thermo_values["Rock"]],
                [241e3, *thermo_values["Ice VI"]],
                [250e3, 1200, *thermo_values["Water"][1:]],
                [100e3, *thermo_values["Ice I"]],
            ]
        ),
        "Pure iron inner core": create_layers(
            [
                [300e3, *thermo_values["Iron"]],
                [1771e3, *thermo_values["Rock"]],
                [116e3, *thermo_values["Ice VI"]],
                [117e3, *thermo_values["Ice V"]],
                [125e3, *thermo_values["Ice II"]],
                [146e3, *thermo_values["Ice I"]],
            ]
        ),
    },
    "Grasset": {
        "Core": create_layers(
            [
                [0, 910e3, *thermo_values["Iron"]],
                [910e3, 1710e3, *thermo_values["Silicates"]],
                [1710e3, 2200e3, *thermo_values["HP ices"]],
                [2200e3, 2500e3, *thermo_values["Water"]],
                [2500e3, 2575e3, *thermo_values["Ice I"]],
            ]
        ),
        "No core": create_layers(
            [
                [0, 1870e3, *thermo_values["Silicates"]],
                [1870e3, 2200e3, *thermo_values["HP ices"]],
                [2200e3, 2500e3, *thermo_values["Water"]],
                [2500e3, 2575e3, *thermo_values["Ice I"]],
            ]
        ),
    },
    "Kronrod": create_layers(
        [
            [690e3, *thermo_values["Fe-Si"]],
            [0, *thermo_values["Ice VII"]],
            [0, *thermo_values["Ice VI"]],
            [0, *thermo_values["Ice V"]],
            [280e3, *thermo_values["Water"]],
            [100e3, *thermo_values["Ice I"]],
        ]
    ),
}


def iterate_layers(layers, use_ref_temp=False, max_iterations=100) -> np.ndarray:
    converged = False
    iterations = 0
    values = None

    while not converged:
        new_values = integrate(layers, values=values)

        if iterations == 0 and use_ref_temp:
            Ts = make_temperature_profile(
                new_values[:, :, 2].flatten(),
                num_steps=new_values.shape[0] * new_values.shape[1],
            ).reshape(new_values.shape[0], -1)
            new_values[:, :, 5] = Ts

        new_values = integrate_density_and_temp(layers, new_values, T0=93.6)

        if values is not None:
            diff = new_values - values
            print(diff.sum())
            converged = np.all(np.abs(diff) < 1e-10) and np.sum(diff**2) < 1e-12

        iterations += 1
        values = new_values

    print(
        f"Converged after {iterations=}"
        if converged
        else f"Failed to converge after {iterations=}"
    )

    return np.array(values)


def make_temperature_profile(gs, ocean_range=[2200e3, 2500e3], T0=93, num_steps=10_000):
    alpha = 3e-4
    Cp = 4180

    Ts = np.zeros(num_steps)
    Ts[0] = T0
    dr = (2575e3 - 0) / (num_steps - 1)
    r = 2575e3

    for i in range(1, num_steps):
        if r >= ocean_range[1]:
            Ts[i] = Ts[i - 1] + 8 / 1000 * dr

        elif r < ocean_range[1] and r <= ocean_range[0]:
            Ts[i] = Ts[i - 1] + 0.1 / 1000 * dr

        else:
            Ts[i] = Ts[i - 1] + Ts[i - 1] * alpha * gs[num_steps - i - 2] / Cp * dr

        r -= dr

    return Ts


def verify_results(m_total, g_surface, p_center, MoI_computed):
    G = 6.67408e-11

    # Titan values
    M = 1.3452e23
    R = 2575e3
    g = 1.35
    MoI = 0.352
    pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)

    print(f"Total mass: {m_total:.2e} kg")
    print(f"Pressure at center: {p_center/1e9:.2f} GPa")
    print(f"Gravity at surface: {g_surface:.2f} m/s^2")
    print("-------")

    # Verification
    pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)
    print(f"Moment of inertia error: {abs(MoI_computed - MoI)/MoI * 100:.2f}%")
    print(f"Pressure error: {abs(p_center - pc_theoretical)/pc_theoretical * 100:.2f}%")
    print(f"Mass error (theory): {abs(m_total - M)/M * 100:.2f}%")
    print(f"Gravity error: {abs(g_surface - g)/g * 100:.2f}%")


layers = models["Fortes"]["Dense ocean"]
values = iterate_layers(layers, max_iterations=5)

rs = values[:, :, 0].flatten()
ms = values[:, :, 1].flatten()
gs = values[:, :, 2].flatten()
ps = values[:, :, 3].flatten()
rhos = values[:, :, 4].flatten()
Ts = values[:, :, 5].flatten()

m_total = ms[-1]
p_center = ps[0]
g_surface = gs[-1]
MoI_computed = compute_moment_of_inertia(values) / (m_total * rs[-1] ** 2)
verify_results(m_total, g_surface, p_center, MoI_computed)

with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=5, figsize=(10, 5))

    axes[0].plot(ms, rs / 1e3, color=pal[0], lw=2)  # type: ignore
    axes[1].plot(gs, rs / 1e3, color=pal[1], lw=2)  # type: ignore
    axes[2].plot(ps / 1e9, rs / 1e3, color=pal[2], lw=2)  # type: ignore
    axes[3].plot(rhos, rs / 1e3, color=pal[3], lw=2)  # type: ignore
    axes[4].plot(Ts, rs / 1e3, color=pal[4], lw=2)  # type: ignore

    axes[0].set_ylabel("Radius [km]")
    axes[0].set_xlabel("Mass [kg]")
    axes[1].set_xlabel("Gravity [m/s^2]")
    axes[2].set_xlabel("Pressure [GPa]")
    axes[3].set_xlabel("Density [kg/m^3]")
    axes[4].set_xlabel("Temperature [K]")

    num_layers = values.shape[0]
    if num_layers > 1:
        for i in range(num_layers - 1):
            for ax in axes:
                ax.axhline(
                    layers[i + 1][0] / 1e3,
                    color="k",
                    alpha=0.5,
                    dashes=(5, 5),
                    ls="--",
                    lw=1,
                    zorder=1,
                )

    for i in range(len(axes)):
        if i > 0:
            axes[i].set_yticklabels([])

    plt.tight_layout()
    plt.show()

# %%
# Values: [layer, entry, variable]
# variables = [r,m,g,p,rho,T]
temp = np.ones([len(values[0, :, 0])])

norm_rad_1 = (values[0, :, 0] - (min(values[0, :, 0]) * temp)) / (
    (max(values[0, :, 0]) - min(values[0, :, 0])) * temp
)
norm_rad_2 = (values[1, :, 0] - (min(values[1, :, 0]) * temp)) / (
    (max(values[1, :, 0]) - min(values[1, :, 0])) * temp
)
norm_rad_3 = (values[2, :, 0] - (min(values[2, :, 0]) * temp)) / (
    (max(values[2, :, 0]) - min(values[2, :, 0])) * temp
)
norm_rad_4 = (values[3, :, 0] - (min(values[3, :, 0]) * temp)) / (
    (max(values[3, :, 0]) - min(values[3, :, 0])) * temp
)
# norm_rad_1 = (values[4,:,0] - (min(values[4,:,0])*temp)) / ((max(values[4,:,0]-min(values[4,:,0]))*temp)

norm_temp_1 = (values[0, :, 5] - (min(values[0, :, 5]) * temp)) / (
    (max(values[0, :, 5]) - min(values[0, :, 5])) * temp
)
norm_temp_2 = (values[1, :, 5] - (min(values[1, :, 5]) * temp)) / (
    (max(values[1, :, 5]) - min(values[1, :, 5])) * temp
)
norm_temp_3 = (values[2, :, 5] - (min(values[2, :, 5]) * temp)) / (
    (max(values[2, :, 5]) - min(values[2, :, 5])) * temp
)
norm_temp_4 = (values[3, :, 5] - (min(values[3, :, 5]) * temp)) / (
    (max(values[3, :, 5]) - min(values[3, :, 5])) * temp
)


# norm_val_1 = (values[0,:,[0,5]] - (min(values[0,:,[0,5]])*temp).T) / (min(values[0,:,[0,5]])*temp).T
# norm_val_2 = (values[1,:,[0,5]] - (min(values[1,:,[0,5]])*temp).T) / (min(values[1,:,[0,5]])*temp).T
# norm_val_3 = (values[2,:,[0,5]] - (min(values[2,:,[0,5]])*temp).T) / (min(values[2,:,[0,5]])*temp).T
# norm_val_4 = (values[3,:,[0,5]] - (min(values[3,:,[0,5]])*temp).T) / (min(values[3,:,[0,5]])*temp).T
# norm_val_5 = (values[4,:,[0,5]] - (min(values[4,:,[0,5]])*temp).T) / (min(values[4,:,[0,5]])*temp).T


with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=4, figsize=(10, 5))

    axes[0].plot(norm_temp_1, norm_rad_1, color=pal[0], lw=2)
    axes[1].plot(norm_temp_2, norm_rad_2, lw=2)
    axes[2].plot(norm_temp_3, norm_rad_3, color=pal[2], lw=2)
    axes[3].plot(norm_temp_4, norm_rad_4, color=pal[3], lw=2)
    # axes[4].plot(norm_temp_5, norm_rad_5, color=pal[4], lw=2)

    axes[0].set_ylabel("Normalised depth")
    axes[0].set_xlabel("Normalised temperature [1]")
    axes[1].set_xlabel("Normalised temperature [2]")
    axes[2].set_xlabel("Normalised temperature [3]")
    axes[3].set_xlabel("Normalised temperature [4]")
    # axes[4].set_xlabel("Normalised temperature [5]")

    axes[0].set_xlim(min(norm_temp_1), max(norm_temp_1))
    axes[1].set_xlim(min(norm_temp_2), max(norm_temp_2))
    axes[2].set_xlim(min(norm_temp_3), max(norm_temp_3))
    axes[3].set_xlim(min(norm_temp_4), max(norm_temp_4))
    # axes[4].set_xlim(min(norm_temp_5),max(norm_temp_5))

    axes[0].set_ylim(min(norm_rad_1), max(norm_rad_1))
    axes[1].set_ylim(min(norm_rad_2), max(norm_rad_2))
    axes[2].set_ylim(min(norm_rad_3), max(norm_rad_3))
    axes[3].set_ylim(min(norm_rad_4), max(norm_rad_4))
    # axes[4].set_ylim(min(norm_rad_5),max(norm_rad_5))

    plt.tight_layout()
    plt.show()

# %%
