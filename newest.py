# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import (
    compute_moment_of_inertia,
    create_layers,
    iterate_layers,
    verify_results,
)

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


# %%
def make_temperature_profile(layers, gradients, T0=93, num_steps=10_000):
    dTs = []

    # Starting from innermost layer, top to bottom
    for i, layer in enumerate(layers):
        drs = np.array(
            [
                0,
                *np.diff(
                    np.linspace(
                        layer[0],
                        layer[1],
                        num=num_steps,
                    )
                ),
            ]
        )
        dTs.append(drs * gradients[i] / 1e3)

    # Flip to make it start from topmost layer, top to bottom
    Ts = np.array(dTs)[::-1]
    Ts = np.cumsum(Ts.flatten()).reshape(Ts.shape)
    # Flip both axes to make it start from innermost layer, bottom to top (which burnman wants)
    Ts = np.array([x[::-1] + T0 for x in Ts][::-1])
    return Ts


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


# %%
layers = models["Fortes"]["Dense ocean"]

num_steps = 10_000
gradients = [
    0.1,
    0.1,
    0.1,
    4,
]
ref_temps = make_temperature_profile(layers, gradients, num_steps=num_steps)

values = iterate_layers(
    layers,
    ref_temps=ref_temps,
    max_iterations=5,
    num_steps=num_steps,
)

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

# %%
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

# %%
