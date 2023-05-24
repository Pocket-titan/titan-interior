# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import (
    compute_moment_of_inertia,
    integrate,
    models,
    picks,
    verify_results,
)

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
        # plt.show()
        # plt.savefig(f"./figures/model1_{paper}_{model}.pdf".replace(r"%", ""), dpi=300)


# %%
