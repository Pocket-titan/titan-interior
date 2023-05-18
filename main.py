# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tools import (
    compute_mass,
    compute_moment_of_inertia,
    create_layers,
    integrate_layers,
    integrate_density
)

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

G = 6.6743e-11


# %%
# Titan values
M = 1.3452e23
R = 2575e3
g = 1.35
MoI = 0.352

# Layer models from papers
# Grasset, 1998 (same as on Brightspace forum for reference)
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

# Fortes et al. 2012
# Baseline 2-layer ice shell over rocky core
layers = create_layers(
    [
        [2054e3, 2584.1],
        [132e3, 1346],
        [117e3, 1268.7],
        [125e3, 1193.3],
        [146e3, 932.8],
    ]
)
# Methane clathrate
layers = create_layers(
    [
        # "Rock"
        [2157e3, 2525.3],
        # MH-I
        [418e3, 962.4],
    ]
)
# Pure water-ice model
layers = create_layers(
    [
        [2054e3, 2584.1],
        [132e3, 1346.0],
        [117e3, 1268.7],
        [125e3, 1193.3],
        [146e3, 932.8],
    ]
)
# Light-ocean
layers = create_layers(
    [
        [2116e3, 2542.3],
        [47e3, 1338.9],
        [62e3, 1272.7],
        [250e3, 1023.5],
        [100e3, 930.9],
    ]
)
# Dense ocean
layers = create_layers(
    [
        [1984e3, 2650.4],
        [241e3, 1350.9],
        [250e3, 1281.3],
        [100e3, 930.9],
    ]
)
# Iron sulphide inner core
layers = create_layers(
    [
        [300e3, 5541.2],
        [1775e3, 2536.9],
        [112e3, 1344.5],
        [117e3, 1268.6],
        [125e3, 1193.2],
        [146e3, 932.8],
    ]
)
# Pure iron inner core
# layers = create_layers(
#     [
#         [300, 8058.3],
#         [1771, 2536.6],
#         [116, 1344.8],
#         [117, 1268.6],
#         [125, 1193.2],
#         [146, 932.8],
#     ]
# )


# Earth values
# M = 5.972e24
# R = 6371e3
# g = 9.81
# MoI = 0.3308

# layers = [
#     [0, R, 5515],
# ]

# Sohl et al. 2014:

layers = create_layers(
    [
        [2048e3,2550,0,0,0,0],
        [202e3,1300,0,0,0,0],
        [178e3,1070,0,0,0,0],
        [112e3,950,0,0,0,0]
    ]
)

# Density and temperature test layer: (pure iron inner core)

layers = create_layers(
    [
        [300e3, 8058.3, 166.41e9, 900, 1.12e-5, 8e-3, 0],
        [1771e3, 2536.6, 67.27e9,500, 2e-5, 2090, 1],
        [116e3, 1344.5, 17.82e9,200, 240e-6, 2090, 1],
        [117e3, 1268.6, 13.93e9,150, 240e-6, 2030, 1],
        [125e3, 1193.2, 13.951e9,125, 260e-6, 1000, 1], 
        [146e3, 932.8, 10.995e9,93, 125e-6, 8e-3, 0]
    ]
)#Data from Fortes 2012 and Noya et al 2007. Layer 5 is Antigorite, approximately what Fortes calls 'Rock'

#TODO: adiabatic temperature propagation, propagate density changes

values = integrate_layers(layers)
rs = values[:, :, 0].flatten()
ms = values[:, :, 1].flatten()
gs = values[:, :, 2].flatten()
ps = values[:, :, 3].flatten()



#Testing out new function:

new_values = integrate_density(layers,values,1000)
rs = new_values[:, :, 0].flatten()
ms = new_values[:, :, 1].flatten()
gs = new_values[:, :, 2].flatten()
ps = new_values[:, :, 3].flatten()

m_total = ms[-1]
p_center = ps[0]
g_surface = gs[-1]
print(f"Total mass: {m_total:.2e} kg")
print(f"Pressure at center: {p_center/1e9:.2f} GPa")
print(f"Gravity at surface: {g_surface:.2f} m/s^2")
print("-------")

# Verification
MoI_computed = compute_moment_of_inertia(layers) / (m_total * rs[-1] ** 2)
M_theoretical = compute_mass(layers)
pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)
print(f"Moment of inertia error: {abs(MoI_computed - MoI)/MoI * 100:.2f}%")
print(f"Pressure error: {abs(p_center - pc_theoretical)/pc_theoretical * 100:.2f}%")
print(f"Mass error (theory): {abs(m_total - M_theoretical)/M_theoretical * 100:.2f}%")
print(f"Mass error (numerical): {abs(m_total - M)/M * 100:.2f}%")
print(f"Gravity error: {abs(g_surface - g)/g * 100:.2f}%")

with plt.rc_context({"axes.grid": False}):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 6))

    axes[0].plot(ms, rs / 1e3, color=pal[0], lw=2)  # type: ignore
    axes[1].plot(gs, rs / 1e3, color=pal[1], lw=2)  # type: ignore
    axes[2].plot(ps / 1e9, rs / 1e3, color=pal[2], lw=2)  # type: ignore

    axes[0].set_ylabel("Radius [km]")
    axes[0].set_xlabel("Mass [kg]")
    axes[1].set_xlabel("Gravity [m/s^2]")
    axes[2].set_xlabel("Pressure [GPa]")

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
#Rough plot
rs = new_values[:, :, 0].flatten()
ms = new_values[:, :, 1].flatten()
gs = new_values[:, :, 2].flatten()
ps = new_values[:, :, 3].flatten()
ts = new_values[:,:,4].flatten()
rhos = new_values[:,:,5].flatten()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)

ax1.plot(ts,rs)
ax2.plot(rhos,rs)



plt.show()
