# %%
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seafreeze.seafreeze as sf
from burnman import (
    Composite,
    Layer,
    Mineral,
    Planet,
    minerals,
)
from burnman.tools.chemistry import dictionarize_formula, formula_mass

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

G = 6.67408e-11

# Titan values
M = 1.3452e23
R = 2575e3
g = 1.35
MoI = 0.352
pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)


# %%
def get_ice_values(phase: str):
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
        **minerals.HP_2011_ds62.h2oL().params,
        # Standard stuff
        "name": f"Ice {phase}",
        # "equation_of_state": "bm2",
        # "equation_of_state": "mgd2",
        "molar_mass": molar_mass,
        "n": sum(water_formula.values()),
        # # Seafreeze values
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
    return (mineral, alpha, Kt, Cp)


# %%
materials = {
    "Rock": Composite(
        [
            minerals.JH_2015.olivine([0.7, 0.3]),
            minerals.HP_2011_ds62.atg(),
        ],
        [0.5, 0.5],
    ),
    "Ice I": get_ice_values("Ih")[0],
    "Ice II": get_ice_values("II")[0],
    "Ice V": get_ice_values("V")[0],
    "Ice VI": get_ice_values("VI")[0],
    "Water": minerals.HP_2011_ds62.h2oL(),
    "Iron": minerals.SE_2015.liquid_iron(),
    "Silicates": minerals.JH_2015.olivine([0.7, 0.3]),  # 100% olivine
    "HP ices": get_ice_values("VI")[0],  # density adjustment: volume!
}


def create_burnman_layers(layers, num=21):
    _layers = []
    rs = np.array([0, *np.cumsum([x[0] for x in layers])])

    for i, layer in enumerate(layers):
        _, kind = layer
        assert kind in materials.keys()

        material = materials[kind]
        options = {} if i < len(layers) - 1 else {"temperature_top": 93.6}
        layer = Layer(name=kind, radii=np.linspace(rs[i], rs[i + 1], num=num))
        layer.set_temperature_mode("adiabatic", **options)
        layer.set_material(material)
        _layers.append(layer)

    return _layers


models = {
    "Fortes": {
        "Dense ocean": create_burnman_layers(
            [
                [1984e3, "Rock"],
                [241e3, "Ice VI"],
                [250e3, "Water"],
                [100e3, "Ice I"],
            ],
        ),
        "Pure water ice": create_burnman_layers(
            [
                [2054e3, "Rock"],
                [132e3, "Ice VI"],
                [117e3, "Ice V"],
                [125e3, "Ice II"],
                [146e3, "Ice I"],
            ]
        ),
        "Pure iron inner core": create_burnman_layers(
            [
                [300e3, "Iron"],
                [1771e3, "Rock"],
                [116e3, "Ice VI"],
                [117e3, "Ice V"],
                [125e3, "Ice II"],
                [146e3, "Ice I"],
            ]
        ),
    },
    "Grasset": {
        "Core": create_burnman_layers(
            [
                [910e3, "Iron"],
                [800e3, "Silicates"],
                [490e3, "HP ices"],
                [300e3, "Water"],
                [75e3, "Ice I"],
            ]
        ),
        "No core": create_burnman_layers(
            [
                [1870e3, "Silicates"],
                [330e3, "HP ices"],
                [300e3, "Water"],
                [75e3, "Ice I"],
            ]
        ),
    },
}

# Make the planet
layers = models["Grasset"]["No core"]
titan = Planet("Titan", layers, verbose=True)
titan.make()

print(
    dedent(
        f"""
    M = {titan.mass:.3e} kg
    I = {titan.moment_of_inertia_factor:.4f}
    r = {titan.radius_planet/1e3:.3e} km
    g = {titan.gravity[-1]:.5f} m/s^2
    pc = {titan.pressures[0]/1e9:.4f} GPa

    Errors:
    M = {100 * (titan.mass - M)/M:.2f} %
    I = {100 * (titan.moment_of_inertia_factor - MoI)/MoI:.2f} %
    r = {100 * (titan.radius_planet - R)/R:.2f} %
    g = {100 * (titan.gravity[-1] - g)/g:.2f} %
    pc = {100 * (titan.pressures[0] - pc_theoretical)/pc_theoretical:.2f} %
    """
    )
)

# TODO:
# 1) impose temp gradient in pt 2 (dT/dz = set)
# 2) impose temp gradient in burnman
# 3) convection
# 4) convection in burnman

rs = titan.radii
ms = np.array(
    [0, *np.cumsum(4 * np.pi * titan.density[:-1] * np.diff(titan.radii**3) / 3)]
)
gs = titan.gravity
ps = titan.pressures
rhos = titan.density
Ts = titan.temperatures

m_total = titan.mass
p_center = ps[0]
g_surface = gs[-1]
MoI_computed = titan.moment_of_inertia_factor

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
