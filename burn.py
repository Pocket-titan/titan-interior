# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from burnman import (
    BoundaryLayerPerturbation,
    Composite,
    Composition,
    Layer,
    Mineral,
    Planet,
    Solution,
    minerals,
)
from utils import adjust_mineral_density, get_ice_values, verify_results

np.set_printoptions(precision=4, suppress=True)

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
fesi = minerals.RS_2014_liquids.Fe2SiO4_liquid()
fesi.set_state(5e9, 700)
fesi.evaluate(["rho", "alpha", "C_p", "K_T"], 5e9, 700)
# 3 scenarios:

# %%


# %%
# Make the planet
layers = models["Kronrod"]["15% Fe core"]
titan = Planet("Titan", layers, verbose=True)


# %%
def make_temperature_profile(layers, gradients, T0=93):
    dTs = []

    # Starting from innermost layer, top to bottom
    for i, layer in enumerate(layers):
        drs = np.array(
            [
                0,
                *np.diff(
                    np.linspace(
                        layer.radii[0],
                        layer.radii[-1],
                        num=layer.radii.shape[0],
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


# Inside to outside layers: start with core
gradients = [
    0.1,
    8,
    0.1,
    8,
]
Ts = make_temperature_profile(layers, gradients)

# for i, layer in enumerate(layers):
#     layer.set_temperature_mode("user-defined", Ts[i])


titan.make()


# %%
def rayleigh_number(rho, alpha, dT, g0, D, visc, diff):
    return rho * alpha * g0 * dT * D**3 / (visc * diff)


# %%


# T of bottom of convecting layer
idx, convecting_layer = [[i, x] for i, x in enumerate(titan.layers) if "water" in x.name.lower()][0]
lower_layer = titan.layers[idx - 1]
upper_layer = titan.layers[idx + 1]
T_top = upper_layer.temperatures[0]
# T_bottom = T_top + 0.0005440344841147866
# T_range = [T_top, T_bottom]

Ra, dT = calculate_rayleigh_number_and_dT(1, convecting_layer.radii[-1] - convecting_layer.radii[0])
print(f"Rayleigh number: {Ra:.4e}")
# upper_layer.pressures[0]

perturbation = BoundaryLayerPerturbation(
    convecting_layer.radii[0],
    convecting_layer.radii[-1],
    rayleigh_number=Ra,
    # top has jump of 60K, botttom has jump of 100 - 60 = 40K
    temperature_change=dT,  # 100
    boundary_layer_ratio=(dT / 2) / dT,  # 60 / 100
)


convecting_layer.set_temperature_mode(
    "perturbed-adiabatic",
    perturbation.temperature(convecting_layer.radii),
)

titan.make()


# %%

rs = titan.radii
ms = np.array([0, *np.cumsum(4 * np.pi * titan.density[:-1] * np.diff(titan.radii**3) / 3)])
gs = titan.gravity
ps = titan.pressures
rhos = titan.density
Ts = titan.temperatures

m_total = titan.mass
p_center = ps[0]
g_surface = gs[-1]
MoI_computed = titan.moment_of_inertia_factor

verify_results(m_total, g_surface, p_center, MoI_computed)

with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=5, figsize=(10, 5))

    axes[0].plot(ms, rs / 1e3, color=pal[0], lw=2)
    axes[1].plot(gs, rs / 1e3, color=pal[1], lw=2)
    axes[2].plot(ps / 1e9, rs / 1e3, color=pal[2], lw=2)
    axes[3].plot(rhos, rs / 1e3, color=pal[3], lw=2)
    axes[4].plot(Ts, rs / 1e3, color=pal[4], lw=2)

    axes[0].set_ylabel("Radius [km]")
    axes[0].set_xlabel("Mass [kg]")
    axes[1].set_xlabel("Gravity [m/s^2]")
    axes[2].set_xlabel("Pressure [GPa]")
    axes[3].set_xlabel("Density [kg/m^3]")
    axes[4].set_xlabel("Temperature [K]")

    num_layers = len(titan.layers)
    if num_layers > 1:
        for i in range(num_layers - 1):
            for ax in axes:
                ax.axhline(
                    titan.layers[i + 1].radii[0] / 1e3,
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


# %%
