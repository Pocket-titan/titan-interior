# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import (
    compute_moment_of_inertia,
    get_temp_info,
    iterate_layers,
    layer_map,
    models,
    table,
    thermo_values,
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
# fmt: off
data = np.array(
    [
        [r"\textbf{Ice $I_h$ shell}", 0, 0, r"\textbf{Ice II}", 0, 0, r"\textbf{Ice V}", 0, 0, r"\textbf{Ice VI}", 0, 0, r'\textbf{"Rock"}', 0, 0],
        [r"\textbf{Ice $I_h$ shell}", 0, 0, r"\textbf{Ocean}", 0, 0, r"\textbf{Ice V}", 0, 0, r"\textbf{Ice VI}", 0, 0, r'\textbf{"Rock"}', 0, 0],
        [r"\textbf{Ice $I_h$ shell}", 0, 0, r"\textbf{Ocean}", 0, 0, r"\textbf{Rock-ice mantle}", 0, 0, r"\textbf{Fe-Si core}", 0, 0, "", 0, 0],
        [r"\textbf{Ice $I_h$ shell}", 0, 0, r"\textbf{Ocean}", 0, 0, r"\textbf{HP ices}", 0, 0, r"\textbf{Silicates}", 0, 0, "", 0, 0],
    ]
)
# fmt: on

# %%
picks = [
    ["Fortes", r"Pure water ice"],
    ["Fortes", r"Light ocean"],
    ["Kronrod", r"15% Fe core"],
    ["Grasset", r"No core"],
]

for pick in picks:
    paper, model = pick
    layers = models[paper][model]

    idx = picks.index(pick)
    radii = [f"{a:.0f} - {b:.0f}" for a, b in layers[:, :2] / 1e3]
    data[idx, 1 : (1 + 3 * len(radii)) : 3] = radii[::-1]

    temp_info = get_temp_info(paper, model)

    values = iterate_layers(layers, temp_info)

    def rayleigh_number(rho, alpha, dT, g0, D, visc, diff):
        return rho * alpha * g0 * dT * D**3 / (visc * diff)

    for i, layer in enumerate(layers):
        j = len(layers) - i - 1
        [rs, gs, ps, rhos, Ts] = values[j, ::-1, [0, 2, 3, -2, -1]]
        print(rs[[0, -1]]/1e3)
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
        data[idx, 2 + 3 * i] = f"{Ra:.2e}"

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
        fig, axes = plt.subplots(ncols=5, figsize=(11, 5))

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
            + r"\_".join(model.replace(r"%", r"").split(" "))
            + "}$]"
        )
        plt.tight_layout()
        plt.savefig(f"./figures/model3_{paper}_{model}.pdf", dpi=300)

# %%
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

# %%
