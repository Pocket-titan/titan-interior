import numpy as np
import pandas as pd
import seafreeze.seafreeze as sf
from burnman import Composite, Layer, Mineral, minerals
from burnman.tools.chemistry import dictionarize_formula, formula_mass
from utils import create_layers


def K(K0, C, D):
    return lambda T: K0 + C * T + D * T**2


# Thermoelastic properties of ice VII and its high-pressure polymorphs: Implications for dynamics
# of cold slab subduction in the lower mantle
def iceVII(P, T):
    Ks = 4.02 + 8.514 * P - 0.0812 * P**2 + 5.24 * 1e-4 * P**3
    Cp = 3.32 + 22.11 * np.exp(-0.058 * P)

    K0 = 23.9e9
    K0prime = 4.2
    a0 = -3.9e-4
    a1 = 1.5e-6
    nabla = 0.9

    alpha_0 = a0 + a1 * T
    alpha = alpha_0 * (1 + K0prime / K0 * P) ** (-nabla)

    return alpha, Ks, Cp


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
        lambda T: (1e3 + (289.73 - 0.024015 * T + 131045 / T**2 - 2779 / T ** (1 / 2))) / 2,
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
    # Kronrod, 15% Fe
    "Fe-Si": [
        3700,
        5.9043e-5,  # burnman
        133e9,
        286.70,
    ],
    # Kronrod
    "Ice VII": [
        1600,
        *iceVII(3e6, 300),
    ],
}

rock_ice_avgs = np.sum(
    np.vstack(
        [
            thermo_values["Ice VII"],
            [
                *thermo_values["Ice VI"][:2],
                thermo_values["Ice VI"][2](300),
                *thermo_values["Ice VI"][3:],
            ],
            thermo_values["Ice V"],
        ]
    ).T
    * [0.6, 0.35, 0.05],
    axis=1,
    # 0.60, 0.35, 0.05 ratios of ice
    # VII, VI, V
)

thermo_values["Rock ice mantle"] = [
    2053.5,
    rock_ice_avgs[1],
    54.275e9,
    rock_ice_avgs[3],
]


table = {
    "Water": {
        "visc": 1.8e-6,
        "diff": 1.3e-7,
    },
    "Ice I": {
        "visc": 10e12,  # Glaciers and ice sheets, A. C. Fowler
        "cond": 3.07,  # Thermal Conductivity of the Ice Polymorphsand the Ice Clathrates
    },
    "Ice II": {
        "visc": 4.4e16,  # ice II is very strong! Grain Size–Sensitive Creep in Ice II
        "cond": 2.1,  # Thermal Conductivity of the Ice Polymorphsand the Ice Clathrates
    },
    "Ice V": {
        "visc": 2.8e14,  # VISCOSITY OF ICE V, Sotin
        "cond": 1.5,  # Thermal Conductivity of the Ice Polymorphsand the Ice Clathrates
    },
    "Ice VI": {
        "visc": lambda P: 10**14
        * P,  # Viscosity of high-pressure ice VI and evolution and dynamics of Ganymede
        "cond": 2,  # # Thermal Conductivity of the Ice Polymorphsand the Ice Clathrates
    },
    "Ice VII": {
        "visc": 10e10,  # guessed from phase diagram, RHEOLOGICAL PROPERTIES OF WATER ICE—APPLICATIONS TO SATELLITES OF THE OUTER PLANETS
        "cond": 4.1,  # Thermal Conductivity of the Ice Polymorphsand the Ice Clathrates
    },
    "Fe-Si": {
        "visc": 5e-7,  # at 45% silicon, Viscosity of Fe–Si Melts with Silicon Content up to 45 at %
        "cond": 100,  # Thermal conductivity of Fe-Si alloys and thermal stratification in Earth’s core
    },
    "Rock": {
        "visc": 1.7e13,  # Grasset, Silicates value
        "cond": (3.0 + (0.404 + 0.000246 * 500) ** (-1)) / 2,  # 500K, THERMAL PROPERTIES OF ROCKS
    },
    "Silicates": {"visc": 1.7e13, "diff": 1e-6},  # Grasset  # Grasset
}

table["Rock ice mantle"] = {
    "visc": table["Ice VI"]["visc"](100**6) * 0.6
    + table["Ice VI"]["visc"](10**6) * 0.35
    + table["Ice V"]["visc"] * 0.05,
    "cond": table["Ice VI"]["cond"] * 0.6
    + table["Ice VI"]["cond"] * 0.35
    + table["Ice V"]["cond"] * 0.05,
}

table["HP ices"] = {"visc": table["Ice VI"]["visc"](100**6), "cond": 0.56}  # Grasset


def make_latex_table():
    names = [
        "Ice I",
        "Ice II",
        "Ice V",
        "Ice VI",
        "HP ices",
        "Water",
        "Fe-Si",
        "Rock ice mantle",
        "Rock",
        "Silicates",
    ]

    vals = np.zeros((len(names), 6))

    for name in names:
        rho, alpha, K, Cp = thermo_values[name]
        if callable(Cp):
            Cp = Cp(300)
        if callable(K):
            K = K(300)

        vals[names.index(name), :4] = [rho, alpha, K, Cp]

        visc = table[name]["visc"]
        if callable(visc):
            visc = visc(10**6)
        if "diff" in table[name]:
            diff = table[name]["diff"]
        else:
            cond = table[name]["cond"]
            if callable(cond):
                cond = cond(300)
            diff = cond / (rho * Cp)

        vals[names.index(name), 4:] = [visc, diff]

    df = pd.DataFrame(
        vals,
        columns=[
            r"$\rho_0$ [kg$\cdot m^{-3}$]",
            r"$\alpha$ [$K^{-1}$]",
            r"$K$ [GPa]",
            r"$C_p$ [J$\cdot kg^{-1}\cdot K^{-1}$]",
            r"$\mu$ [Pa$\cdot$ s]]",
            r"$\kappa$ [$m^2\cdot s^{-1}$]",
        ],
        index=names,
    )
    str = df.style.format(
        {
            r"$\rho_0$ [kg$\cdot m^{-3}$]": lambda x: f"{x:.0f}",
            r"$\alpha$ [$K^{-1}$]": lambda x: f"{x:.2e}",
            r"$K$ [GPa]": lambda x: f"{x/1e9:.2f}",
            r"$C_p$ [J$\cdot kg^{-1}\cdot K^{-1}$]": lambda x: f"{x:.0f}",
            r"$\mu$ [Pa$\cdot$ s]]": lambda x: f"{x:.2e}",
            r"$\kappa$ [$m^2\cdot s^{-1}$]": lambda x: f"{x:.2e}",
        },
    ).to_latex(
        hrules=True,
        caption="Thermal and transport properties of layer materials.",
        label="tab:materials",
    )

    print(str)


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
    "Kronrod": {
        "15% Fe core": create_layers(
            [
                [690e3, *thermo_values["Fe-Si"]],
                [1500e3, *thermo_values["Rock ice mantle"]],
                [280e3, *thermo_values["Water"]],
                [100e3, *thermo_values["Ice I"]],
            ]
        ),
    },
}


# Burnman things
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
    "Rock ice mantle": Composite(
        [
            get_ice_values("V")[0],
            get_ice_values("VI", rho_0=1310)[0],
            get_ice_values("VI", rho_0=1600)[0],
        ],
        [0.05, 0.35, 0.60],
    ),
    "Fe-Si": minerals.RS_2014_liquids.Fe2SiO4_liquid(),
}


burnman_models = {
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
    "Kronrod": {
        "15% Fe core": create_burnman_layers(
            [
                [690e3, "Fe-Si"],
                [1500e3, "Rock ice mantle"],
                [280e3, "Water"],
                [100e3, "Ice I"],
            ]
        ),
    },
}
