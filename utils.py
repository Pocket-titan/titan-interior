from textwrap import dedent

import numpy as np
import pandas as pd
import seafreeze.seafreeze as sf
from burnman import Composite, Layer, Mineral, minerals
from burnman.tools.chemistry import dictionarize_formula, formula_mass

picks = [
    ["Fortes", r"Pure water ice"],
    ["Fortes", r"Light ocean"],
    ["Kronrod", r"15% Fe core"],
    ["Grasset", r"No core"],
]


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


# Thermo
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
                # [2054e3, *thermo_values["Rock"]],
                # [132e3, *thermo_values["Ice VI"]],
                # [117e3, *thermo_values["Ice V"]],
                # [125e3, *thermo_values["Ice II"]],
                # [146e3, *thermo_values["Ice I"]],
                [1910e3, *thermo_values["Rock"]],
                [110e3, *thermo_values["Ice VI"]],
                [110e3, *thermo_values["Ice V"]],
                [320e3, *thermo_values["Ice II"]],
                [125e3, *thermo_values["Ice I"]],
            ]
        ),
        "Light ocean": create_layers(
            [
                # [2116e3, *thermo_values["Rock"]],
                # [47e3, *thermo_values["Ice VI"]],
                # [62e3, *thermo_values["Ice V"]],
                # [250e3, 950, *thermo_values["Water"][1:]],
                # [100e3, *thermo_values["Ice I"]],
                [1900e3, *thermo_values["Rock"]],
                [100e3, *thermo_values["Ice VI"]],
                [100e3, *thermo_values["Ice V"]],
                [300e3, 950, *thermo_values["Water"][1:]],
                [175e3, *thermo_values["Ice I"]],
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
                # [0, 1870e3, *thermo_values["Silicates"]],
                # [1870e3, 2200e3, *thermo_values["HP ices"]],
                # [2200e3, 2500e3, *thermo_values["Water"]],
                # [2500e3, 2575e3, *thermo_values["Ice I"]],
                [1800e3, *thermo_values["Silicates"]],
                [500e3, *thermo_values["HP ices"]],
                [200e3, *thermo_values["Water"]],
                [75e3, *thermo_values["Ice I"]],
            ]
        ),
    },
    "Kronrod": {
        "15% Fe core": create_layers(
            [
                # [690e3, *thermo_values["Fe-Si"]],
                # [1500e3, *thermo_values["Rock ice mantle"]],
                # [280e3, *thermo_values["Water"]],
                # [100e3, *thermo_values["Ice I"]],
                [1000e3, *thermo_values["Fe-Si"]],
                [1300e3, *thermo_values["Rock ice mantle"]],
                [200e3, *thermo_values["Water"]],
                [75e3, *thermo_values["Ice I"]],
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
        "Light ocean": create_burnman_layers(
            [
                [1900e3, "Rock"],
                [100e3, "Ice VI"],
                [100e3, "Ice V"],
                [300e3, "Light water"],
                [175e3, "Ice I"],
            ]
        ),
        "Pure water ice": create_burnman_layers(
            [
                [1910e3, "Rock"],
                [110e3, "Ice VI"],
                [110e3, "Ice V"],
                [320e3, "Ice II"],
                [125e3, "Ice I"],
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
                [1800e3,"Silicates"],
                [500e3,"HP ices"],
                [200e3,"Water"],
                [75e3,"Ice I"],
            ]
        ),
    },
    "Kronrod": {
        "15% Fe core": create_burnman_layers(
            [
                [1000e3, "Fe-Si"],
                [1300e3, "Rock ice mantle"],
                [200e3, "Water"],
                [75e3, "Ice I"],
            ]
        ),
    },
}


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


def get_temp_info(paper, model):
    layers = models[paper][model]
    temp_info = []

    # Calculate T contrast over first layer
    d = layers[-1][1] - layers[-1][0]
    q = 5e-3  # observed heat flux, W/m^2, Kalousová and Sotin, 2020
    dT = q * d / table["Ice I"]["cond"]  # Grasset

    # Grasset
    if paper == "Grasset":
        temp_info = [
            {
                "top": 93,
                "gradient": dT / (d / 1e3),
            },
            {"mode": "adiabatic"},
            {"bottom": 300},
            {"mode": "adiabatic"},
        ]

    # Fortes
    if paper == "Fortes":
        temp_info = [
            {
                "top": 93,
                "gradient": dT / (d / 1e3),
            },
            {"mode": "adiabatic"} if model == r"Light ocean" else {"gradient": 0.1},
            {"gradient": 0.1},
            {"gradient": 0.1},
            {"mode": "adiabatic", "bottom": 900},
        ]

        values = iterate_layers(layers, temp_info)
        T_top = values[2, 0, 5]
        values[:, :, 5] = 0.0
        Ts = propagate_adiabatic(
            [layers[0][1], layers[0][0], *layers[0][2:]],
            values[0, ::-1],
            T0=900,
        )
        T_bottom = Ts[-1]
        gradient = (T_bottom - T_top) / (layers[1][1] / 1e3 - layers[1][0] / 1e3)

        temp_info = [
            {
                "top": 93,
                "gradient": dT / (d / 1e3),
            },
            {"mode": "adiabatic"} if model == "Light ocean" else {"gradient": 0.1},
            {"gradient": 0.1},
            {"gradient": gradient},
            {"mode": "adiabatic", "bottom": 900},
        ]

    # Kronrod
    if paper == "Kronrod":
        dT = 273 - 93
        gradient = (380 - 273) / (layers[-1][1] / 1e3)
        temp_info = [
            {
                "top": 93,
                "gradient": dT / (d / 1e3),
            },
            {"gradient": gradient},
            {"gradient": gradient},
            {"gradient": gradient},
        ]

    return temp_info


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


def generate_error_table():
    # fmt: off
    data = [
        # Model 1                   # Model 2                    # Model 3
        [0.63, 70.45, 0.93, 6.96,   0.63, 70.45, 0.93, 6.96,     6.59, 42.66, 6.31, 5.14],
        [4.33, 64.85, 4.04, 9.58,   4.33, 64.85, 4.04, 9.58,     11.24, 37.50, 10.97, 7.59],
        [2.08, 63.05, 1.79, 1.00,   2.08, 63.05, 1.79, 1.00,     10.78, 67.33, 10.51, 0.14],
        [0.76, 92.73, 1.07, 10.10,  0.76, 92.73, 1.07, 10.10,    14.60, 145.82, 14.94, 8.69],
    ]
    # fmt: on

    cols = [r"$M$", r"$P_c$", r"$g_0$", r"$I/MR^2$"]
    index = pd.MultiIndex.from_tuples(picks)
    columns = pd.MultiIndex.from_tuples(
        [(x, y) for x in ["Model 1", "Model 2", "Model 3"] for y in cols]
    )
    df = pd.DataFrame(data, index=index, columns=columns)

    str = df.T.style.format(formatter=lambda x: f"{x:2.2f}%").to_latex(
        hrules=True,
        caption=r"errors",
        label="tab:errors",
    )

    print(str)

def make_models_table(data):
    index = pd.MultiIndex.from_tuples(picks)

    df = pd.DataFrame(
        data,
        columns=[
            "",
            "$r$ [km]",
            "$Ra$ [-]",
            "",
            "$r$ [km]",
            "$Ra$ [-]",
            "",
            "$r$ [km]",
            "$Ra$ [-]",
            "",
            "$r$ [km]",
            "$Ra$ [-]",
            "",
            "$r$ [km]",
            "$Ra$ [-]",
        ],
        index=index,
    )
    print(
        df.T.style.to_latex(
            hrules=True,
            caption=r"Models selected from literature for comparison. See table \ref{tab:materials} for the thermoelastic properties associated with the specific layers.",
            label="tab:layers",
        )
    )
