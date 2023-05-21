# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seafreeze.seafreeze as sf
from burnman import (
    BoundaryLayerPerturbation,
    Composite,
    Layer,
    Mineral,
    Planet,
    minerals,
)
from utils import adjust_mineral_density, get_ice_values, verify_results

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

# 3 scenarios:

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
    "Light water": adjust_mineral_density(minerals.HP_2011_ds62.h2oL(), 950),
    "Dense water": adjust_mineral_density(minerals.HP_2011_ds62.h2oL(), 1200),
    "Iron": minerals.SE_2015.liquid_iron(),
    "Silicates": minerals.JH_2015.olivine([0.7, 0.3]),  # 100% olivine
    "HP ices": get_ice_values("VI", rho_0=1310)[0],
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
                [250e3, "Dense water"],
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
                [910e3, "Iron"],  # Temp gradient 6 K/GPa *Urakawa
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
layers = models["Fortes"]["Dense ocean"]
titan = Planet("Titan", layers, verbose=True)


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
    0.1,
    2,
    0.1,
]
Ts = make_temperature_profile(layers, gradients)

for i, layer in enumerate(layers):
    layer.set_temperature_mode("user-defined", Ts[i])


def calculate_rayleigh_number(convecting_layer, T_range):
    g_avg = np.mean(convecting_layer.gravity)
    r_range = convecting_layer.radii[[0, -1]]
    T_avg = np.mean(T_range)

    alpha = np.mean(convecting_layer.alpha)
    rho = np.mean(convecting_layer.rho)
    Cp = np.mean(convecting_layer.C_p)

    Pr = 10  # turbulent
    # Pr = 1  # turbulent

    # # Vogel-Fulcher-Tammann equation for dynamic viscosity of water:
    # A = 0.02939
    # B = 507.88
    # C = 149.3
    # visc = A * np.exp(B / (T_avg - C)) * 1e-3

    # _rho = 1000
    # visc = 0.0168 * _rho * (20) ** (-0.88)  # 20 C

    # Thermal conductivity of water at 20 deg C (https://pubs.aip.org/aip/jpr/article/41/3/033102/242059/New-International-Formulation-for-the-Thermal)
    k = 0.598
    diff = k / (rho * Cp)
    visc = diff * Pr

    dT = abs(T_range[1] - T_range[0])
    l = abs(r_range[1] - r_range[0])
    l = 100e3
    print(l)
    Ra = rho * alpha * dT * l**3 * g_avg / (diff * visc)

    # Roche et al. (2010)
    # Gastine et al. (2015)
    # Gastine et al. (2016)

    # New
    rho = 1200
    g0 = 1.35  # good!
    alpha = 3.2e-4  # good!
    Cp = 2800
    visc = 1.8e-6  # good!
    diff = 1.3e-7  # good!
    dT = dT = 7.3 * (visc / (alpha * g0 * rho * Cp)) ** (1 / 4) * (3 * 1e-3) ** (3 / 4)
    Ra = rho * alpha * dT * l**3 * g0 / (diff * visc)

    print(Ra)

    return Ra


titan.make()
# %%
# T of bottom of convecting layer
idx, convecting_layer = [[i, x] for i, x in enumerate(titan.layers) if "water" in x.name.lower()][0]
lower_layer = titan.layers[idx - 1]
upper_layer = titan.layers[idx + 1]
T_top = upper_layer.temperatures[0]
T_bottom = T_top + 0.0005440344841147866


T_range = [T_top, T_bottom]

Ra = calculate_rayleigh_number(convecting_layer, T_range)
print(f"Rayleigh number: {Ra:.4e}")
# upper_layer.pressures[0]

dT = T_bottom - T_top

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
Omega = 4.6e-6
rho = 1200
g0 = 1.35  # good!
alpha = 3.2e-4  # good!
Cp = 2800
visc = 1.8e-6  # good!
diff = 1.3e-7  # good!
q0 = 7e-3  # [6-8e-3] mW/m^2

D = 250e3

from scipy.optimize import minimize_scalar, root_scalar


def calculate_nussert_rayleigh(dT):
    Nu = q0 * D / (rho * Cp * diff * dT)
    Ra = alpha * g0 * dT * D**3 / (visc * diff)
    E = visc / (Omega * D**2)

    Nu1 = 0.07 * Ra ** (1 / 3)
    Nu2 = 0.0171 * Ra**0.389
    Nu3 = 0.15 * Ra ** (3 / 2) * E**2

    dNu1 = Nu1 - Nu
    dNu2 = Nu2 - Nu
    dNu3 = Nu3 - Nu

    return (dNu1, dNu2, dNu3, Nu, dT, Ra)


# res = minimize_scalar(calculate_nussert_rayleigh, bounds=(0, 100), method="bounded")
res = root_scalar(lambda x: calculate_nussert_rayleigh(x)[1], x0=1e-7, x1=1e-5)
print(res)
print(calculate_nussert_rayleigh(res.root)[-1])

# %%
