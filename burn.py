# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seafreeze.seafreeze as sf
from burnman import (
    Composite,
    Composition,
    Layer,
    Material,
    Mineral,
    PerplexMaterial,
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

# %%


ice_vii = Mineral(
    {
        "name": "ice_VII",
        "equation_of_state": "bm2",
        "V_0": 12.49e-6,
        "K_0": 20.15e9,
        "Kprime_0": 4.0,
        "molar_mass": 0.01801528,
    }
)  # https://github.com/CaymanUnterborn/ExoPlex/blob/6d4d80a9cec5c90732b3a0900c0fcba7183fff09/Examples/make_water_grid.py#L8

# water = Mineral(
#     {
#         "equation_of_state": "bm2",
#         "V_0": 18.797e-6,
#         "K_0": 2.06e9,
#         "Kprime_0": 6.29,
#         "molar_mass": 0.01801528,
#         "Kprime_prime_0": (-1.89 / 2.06e9),
#         "n": 1,
#     }
# )  #


# liquid_iron = minerals.SE_2015.liquid_iron()


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

    params = {
        # Standard stuff
        "name": f"Ice {phase}",
        "equation_of_state": "bm2",
        "molar_mass": molar_mass,
        "n": sum(water_formula.values()),
        # Seafreeze values
        "K_0": out.Kt * 1e6,
        "Kprime_0": out.Kp,
        "G_0": out.shear * 1e6,
        "V_0": out.V * molar_mass,
        "H_0": out.H,
        "S_0": out.S,
        "P_0": ref_p,
        "T_0": ref_T,
        "grueneisen_0": out.alpha * out.Kt * 1e6 / (out.Cv * out.rho),
        "Cp": [out.Cp, 0, 0, 0],
        "Cv": [out.Cv, 0, 0, 0],
    }

    mineral = Mineral(params=params)
    return (mineral, out.alpha, out.Kt * 1e3, out.Cp)


# %%
# antigorite = Composition(
#     {
#         "MgO": 30.15,
#         "FeO": 17.92,
#         "SiO2": 39.95,
#         "H2O": 11.98,
#     },
#     "weight",
#     normalize=True,
# )  # http://webmineral.com/data/Antigorite.shtml
rock = Composite(
    [
        minerals.JH_2015.olivine([0.7, 0.3]),
        minerals.HP_2011_ds62.atg(),
    ],
    [0.5, 0.5],
)
ice_vi_material = get_ice_values("VI")[0]
liquid_water = minerals.HP_2011_ds62.h2oL()
ice_i_material = get_ice_values("Ih")[0]

ts = np.array([99.96, 96.51, 94.71, 93.6])
ps = np.array([1.09e9, 5.87e8, 1.36e8, 1.34e4])
gs = np.array([0.0, 1.64, 1.54, 1.47])

# Layers
core = Layer(name="Rock", radii=np.linspace(0, 1984e3, num=100))
core.set_material(rock)
core.set_temperature_mode("adiabatic", temperature_top=ts[0])
core.set_pressure_mode("self-consistent", pressure_top=ps[0], gravity_bottom=gs[0])
core.make()

core.temperatures, core.pressures
# %%

ice_vi = Layer(name="Ice VI", radii=np.linspace(1984e3, 2225e3, num=100))
ice_vi.set_material(ice_vi_material)
ice_vi.set_temperature_mode("adiabatic", temperature_top=ts[1])
ice_vi.set_pressure_mode("self-consistent", pressure_top=ps[1], gravity_bottom=gs[1])
# ice_vi.make()

ocean = Layer(name="Ocean", radii=np.linspace(2225e3, 2475e3, num=100))
ocean.set_material(liquid_water)
ocean.set_temperature_mode("adiabatic", temperature_top=ts[2])
ocean.set_pressure_mode("self-consistent", pressure_top=ps[2], gravity_bottom=gs[2])

# %%
ice_i = Layer(name="Ice I", radii=np.linspace(2475e3, 2575e3, num=100))
ice_i.set_material(ice_i_material)
ice_i.set_temperature_mode("adiabatic", temperature_top=ts[3])
ice_i.set_pressure_mode("self-consistent", pressure_top=ps[3], gravity_bottom=gs[3])
# ice_i.make()

rock.C_p

# titan = Planet("Titan", [core, ice_vi, ocean, ice_i], verbose=True)
# titan.make()
# "Dense ocean": create_layers(
#     [
#         [1984e3, *thermo_values["Rock"]],
#         [241e3, *thermo_values["Ice VI"]],
#         [250e3, 1200, *thermo_values["Water"][1:]],
#         [100e3, *thermo_values["Ice I"]],
#     ]
# ),

# %%
ps[0]
# %%
