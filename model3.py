# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from thermo import models, table, thermo_values
from utils import (
    compute_moment_of_inertia,
    integrate,
    integrate_density,
    iterate_layers,
    layer_map,
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


# %%
paper, model = ["Grasset", r"No core"]
layers = models[paper][model]

# top of silicates is 300K
# Silicates ~ 1400 K, convective

d = layers[-1][1] - layers[-1][0]
q = 5e-3  # W/m^2, Kalousová and Sotin, 2020
dT = q * d / table["Ice I"]["cond"]  # Grasset

temp_info = [
    {
        "top": 93,
        "gradient": dT / (d / 1e3),
    },
    {"mode": "adiabatic"},
    {"bottom": 300},
    {"mode": "adiabatic"},
]

iterate_layers(layers, temp_info)


# %%
values = iterate_layers(
    layers,
    ref_temps=ref_temps,
    num_steps=num_steps,
    max_iterations=100,
)


def rayleigh_number(rho, alpha, dT, g0, D, visc, diff):
    return rho * alpha * g0 * dT * D**3 / (visc * diff)


for i, layer in enumerate(layers):
    j = len(layers) - i - 1
    [rs, gs, ps, rhos, Ts] = values[j, ::-1, [0, 2, 3, -2, -1]]
    D = np.abs(rs[0] - rs[-1])
    rho = np.mean(rhos)
    g0 = np.mean(gs)
    dT = Ts[-1] - Ts[0]

    T_avg = np.mean(Ts)
    p_avg = np.mean(ps)

    name = layer_map[paper][model][j]
    print(name)

    _, alpha, K, Cp = thermo_values[name]
    if callable(Cp):
        Cp = Cp(T_avg)
    if callable(K):
        K = K(T_avg)

    visc = table[name]["visc"]
    if callable(visc):
        visc = visc(p_avg)
    if "diff" in table[name]:
        diff = table[name]["diff"]
    else:
        cond = table[name]["cond"]
        if callable(cond):
            cond = cond(T_avg)
        diff = cond / (rho * Cp)

    Ra = rayleigh_number(rho, alpha, dT, g0, D, visc, diff)
    print(f"Ra = {Ra:.2e}")


rs = values[:, :, 0].flatten()
ms = values[:, :, 1].flatten()
gs = values[:, :, 2].flatten()
ps = values[:, :, 3].flatten()
rhos = values[:, :, 4].flatten()
Ts = values[:, :, 5].flatten()

print(f"{paper}, {model}")
m_total = ms[-1]
p_center = ps[0]
g_surface = gs[-1]
MoI_computed = compute_moment_of_inertia(values) / (m_total * rs[-1] ** 2)
verify_results(m_total, g_surface, p_center, MoI_computed)

with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=5, figsize=(9, 5))

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

    fig.suptitle(
        "Model 3 [$"
        + r"\it{"
        + paper
        + ", "
        + r"\_".join(model.replace(r"%", r"\%").split(" "))
        + "}$]"
    )
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"./figures/model3_{paper}_{model}.pdf", dpi=300)
