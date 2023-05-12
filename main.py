# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tools import integrate_layers

sns.set_theme(style="whitegrid", palette="Set2")
pal = sns.color_palette("Set2")

# %%
# Titan values
R = 2575e3
g = 1.35

# Layers from Grasset, 1998 (same as on Brightspace forum for reference)
layers = [
    # Iron core
    [0, 910e3, 8000],
    # Silicates
    [910e3, 1710e3, 3300],
    # HP ices + hydrates
    [1710e3, 2200e3, 1310],
    # Liquid layer
    [2200e3, 2500e3, 1000],
    # Ice 1 layer
    [2500e3, 2575e3, 917],
]


values = integrate_layers(layers)
rs = values[:, :, 0].flatten()
ms = values[:, :, 1].flatten()
gs = values[:, :, 2].flatten()
ps = values[:, :, 3].flatten()

total_m = ms[-1]
p_center = ps[0]
g_surface = gs[-1]
print(f"Total mass: {total_m:.2e} kg")
# TODO: for some reason the pressure is 1e3 times too large?
print(f"Pressure at center: {p_center/1e6:.2e} MPa")
print(f"Gravity at surface: {g_surface:.2f} m/s^2")

with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 6))

    axes[0].plot(ms, rs / 1e3, color=pal[0], lw=2)
    axes[1].plot(gs, rs / 1e3, color=pal[1], lw=2)
    axes[2].plot(ps / 1e6, rs / 1e3, color=pal[2], lw=2)

    axes[0].set_ylabel("Radius [km]")
    axes[0].set_xlabel("Mass [kg]")
    axes[1].set_xlabel("Gravity [m/s^2]")
    axes[2].set_xlabel("Pressure [MPa]")

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

    plt.tight_layout()
    plt.show()

# %%
