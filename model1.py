# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import (
    compute_moment_of_inertia,
    integrate,
    models,
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
picks = [
    ["Fortes", r"Pure water ice"],
    ["Fortes", r"Light ocean"],
    ["Kronrod", r"15% Fe core"],
    ["Grasset", r"No core"],
]

for pick in picks:
    paper, model = pick
    layers = models[paper][model]

    values = integrate(layers, num_steps=10_000)

    rs = values[:, :, 0].flatten()
    ms = values[:, :, 1].flatten()
    gs = values[:, :, 2].flatten()
    ps = values[:, :, 3].flatten()
    rhos = values[:, :, 4].flatten()

    print(f"{paper}, {model}")
    m_total = ms[-1]
    p_center = ps[0]
    g_surface = gs[-1]
    MoI_computed = compute_moment_of_inertia(values) / (m_total * rs[-1] ** 2)
    verify_results(m_total, g_surface, p_center, MoI_computed)

    with plt.rc_context({"axes.grid": False}):
        fig, axes = plt.subplots(ncols=4, figsize=(9, 5))

        axes[0].plot(ms, rs / 1e3, color=pal[0], lw=2)
        axes[1].plot(gs, rs / 1e3, color=pal[1], lw=2)
        axes[2].plot(ps / 1e9, rs / 1e3, color=pal[2], lw=2)
        axes[3].plot(rhos, rs / 1e3, color=pal[3], lw=2)

        axes[0].set_ylabel("Radius [km]")
        axes[0].set_xlabel("Mass [kg]")
        axes[1].set_xlabel("Gravity [m/s^2]")
        axes[2].set_xlabel("Pressure [GPa]")
        axes[3].set_xlabel("Density [kg/m^3]")

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
            "Model 1 [$"
            + r"\it{"
            + paper
            + ", "
            + r"\_".join(model.replace(r"%", r"\%").split(" "))
            + "}$]"
        )
        plt.tight_layout()
        # plt.savefig(f"./figures/model1_{paper}_{model}.pdf".replace(r"%", ""), dpi=300)

# %%
import pandas as pd

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

# %%
