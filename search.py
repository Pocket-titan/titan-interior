# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from utils import (
    compute_moment_of_inertia,
    create_layers,
    integrate,
    layer_map,
    models,
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

G = 6.67408e-11

# Titan values
M = 1.3452e23
R = 2575e3
g = 1.35
MoI = 0.352
pc_theoretical = 3 * G * M**2 / (8 * np.pi * R**4)


def generate_radii(thickness_ranges):
    thick = [0] * len(thickness_ranges)

    indices = [*range(len(thickness_ranges))]
    while len(indices) > 0:
        i = np.random.randint(0, len(indices))
        index = indices.pop(i)

        if len(indices) == 0:
            thickness = 2575e3 - sum(thick)
        else:
            thickness_range = thickness_ranges[index]
            thickness = np.random.randint(*thickness_range)

        thick[index] = thickness

    radii = np.reshape(
        np.cumsum([[0, x] for x in thick]),
        (len(thick), 2),
    )

    if np.sum([[-a, b] for a, b in radii], axis=1).min() < 0:
        return generate_radii(thickness_ranges)

    return radii


def try_random_radii(thickness_ranges, names, num=1000):
    errors = []
    radiuses = []

    for _ in tqdm(range(num)):
        radii = generate_radii(thickness_ranges)
        layers = create_layers(
            np.hstack(
                [
                    radii,
                    [thermo_values[x] for x in names],
                ]
            )
        )
        values = integrate(layers, num_steps=10_000)

        m_total = values[-1, -1, 1]
        p_center = values[0, 0, 3]
        g_surface = values[-1, -1, 2]
        MoI_computed = compute_moment_of_inertia(values) / (m_total * values[-1, -1, 0] ** 2)

        mass_error = abs(m_total - M) / M * 100
        p_error = abs(p_center - pc_theoretical) / pc_theoretical * 100
        g_error = abs(g_surface - g) / g * 100
        MoI_error = abs(MoI_computed - MoI) / MoI * 100
        errors.append([mass_error, p_error, g_error, MoI_error])
        radiuses.append(radii)

    return np.array(errors), np.array(radiuses)


# %%
picks = [
    ["Fortes", r"Pure water ice"],
    ["Fortes", r"Light ocean"],
    ["Kronrod", r"15% Fe core"],
    ["Grasset", r"No core"],
]

for paper, model in picks:
    base_radii = models[paper][model][:, :2]
    names = layer_map[paper][model]

    base_thick = [b - a for a, b in base_radii]
    thickness_ranges = [[max(0.33 * x, 10e3), min(3 * x, 2400e3)] for x in base_thick]

    errors, radiuses = try_random_radii(
        thickness_ranges,
        names,
        num=100,
    )

    mass_inertia_error = errors[:, [0, 3]].sum(axis=1)
    total_error = errors.sum(axis=1)

    print(paper, model)
    # errors_to_min = total_error
    errors_to_min = mass_inertia_error
    idx = np.argmin(errors_to_min)
    print(errors_to_min[idx])
    print(errors[idx])
    print(radiuses[idx] / 1e3)


# %%
paper, model = picks[1]
names = layer_map[paper][model]

layers = create_layers(
    np.hstack(
        [
            1e3
            * np.array(
                [
                    [0.0, 1900],
                    [1900, 2000],
                    [2000, 2100],
                    [2100, 2400],
                    [2400, 2575.0],
                ]
            ),
            [thermo_values[x] for x in names],
        ]
    )
)
values = integrate(layers)

rs = values[:, :, 0].flatten()
ms = values[:, :, 1].flatten()
gs = values[:, :, 2].flatten()
ps = values[:, :, 3].flatten()
rhos = values[:, :, 4].flatten()

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

    plt.tight_layout()
    plt.show()


# %%
rs = np.array(
    [
        [0, 1800],
        [1800, 2300],
        [2300, 2500],
        [2500, 2575],
    ]
)
drs = rs[:, 1] - rs[:, 0]
print(list([[x, f"q*thermo_values[{layer_map[paper][model][i]}]"] for i, x in enumerate(drs)]))
# %%
