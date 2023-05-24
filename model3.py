# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from burnman import Layer, Planet
from utils import (
    burnman_models,
    get_temp_info,
    iterate_layers,
    layer_map,
    make_models_table,
    models,
    picks,
    table,
    thermo_values,
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


def rayleigh_number(rho, alpha, dT, g0, D, visc, diff):
    return rho * alpha * g0 * dT * D**3 / (visc * diff)


for pick in picks:
    paper, model = pick
    print(paper, model)
    layers: list[Layer] = burnman_models[paper][model]

    model2_layers = models[paper][model]
    temp_info = get_temp_info(paper, model)

    if model == r"Pure water ice":
        temp_info[-1]["top"] = 875

    idx = picks.index(pick)
    radii = [f"{a:.0f} - {b:.0f}" for a, b in model2_layers[:, :2] / 1e3]
    data[idx, 1 : (1 + 3 * len(radii)) : 3] = radii[::-1]

    values = iterate_layers(model2_layers, temp_info)

    for i, layer in enumerate(model2_layers):
        j = len(model2_layers) - i - 1
        [rs, gs, ps, rhos, Ts] = values[j, ::-1, [0, 2, 3, -2, -1]]
        print(rs[[0, -1]] / 1e3)
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

    # below critical Ra, conduction instead of convection
    # ice: 10**4
    # water: 10**8
    # Earths mantle: 10**3

    for i, layer in enumerate(reversed(layers)):
        assert layer.radii is not None
        j = len(layers) - i - 1
        t = temp_info[i]

        top_layer = layers[j + 1] if i > 0 else None
        bot_layer = layers[j - 1] if i < len(layers) - 1 else None

        p_options = {"pressure_mode": "self-consistent"}
        t_options = {"temperature_mode": "adiabatic"}

        if "mode" in t and t["mode"] == "adiabatic":
            temperature_top = t["top"] if "top" in t else top_layer.temperatures[0]
            t_options["temperature_top"] = temperature_top
        else:
            if "bottom" in t and "gradient" not in t:
                temperature_top = top_layer.temperatures[0]
                gradient = (t["bottom"] - temperature_top) / (
                    layer.radii[-1] / 1e3 - layer.radii[0] / 1e3
                )
                rs = layer.radii[::-1] - layer.radii[0]
                Ts = t["bottom"] + gradient * (rs / 1e3)
                t_options["temperature_mode"] = "user-defined"
                t_options["temperatures"] = Ts

        if "gradient" in t:
            Ts = None

            if "top" in t or (
                top_layer is not None
                and hasattr(top_layer, "temperatures")
                and top_layer.temperatures is not None
            ):
                temperature_top = t["top"] if "top" in t else top_layer.temperatures[0]
                rs = (layer.radii[::-1] - layer.radii[0])[::-1]
                Ts = temperature_top + t["gradient"] * (rs / 1e3)
                Ts = Ts[::-1]
            elif "bottom" in t or (
                bot_layer is not None
                and hasattr(bot_layer, "temperatures")
                and bot_layer.temperatures is not None
            ):
                temperature_bottom = t["bottom"] if "bottom" in t else bot_layer.temperatures[-1]
                rs = layer.radii[::-1] - layer.radii[0]
                Ts = temperature_bottom + t["gradient"] * (rs / 1e3)

            if Ts is not None:
                t_options["temperature_mode"] = "user-defined"
                t_options["temperatures"] = Ts
            else:
                print("Ignoring gradient, this is probably not what you want...")

        try:
            p_options["gravity_bottom"] = bot_layer.gravity[-1]
        except:
            p_options["gravity_bottom"] = values[j, 0, 2]

        try:
            p_options["pressure_top"] = top_layer.pressures[0]
        except:
            p_options["pressure_top"] = values[j, -1, 3]

        layer.set_temperature_mode(**t_options)
        layer.set_pressure_mode(**p_options)
        layer.make()

    titan = Planet(name="Titan", layers=layers, verbose=True)
    titan.make()

    rs = titan.radii
    ms = np.array([0, *np.cumsum(4 * np.pi * titan.density[:-1] * np.diff(titan.radii**3) / 3)])
    gs = titan.gravity
    ps = titan.pressures
    rhos = titan.density
    Ts = titan.temperatures

    m_total = titan.mass
    p_center = ps[0]
    g_surface = gs[-1]
    MoI_computed = titan.moment_of_inertia_factor

    verify_results(m_total, g_surface, p_center, MoI_computed)

    with plt.rc_context({"axes.grid": False}):
        fig, axes = plt.subplots(ncols=5, figsize=(10, 5))

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

        num_layers = len(titan.layers)
        if num_layers > 1:
            for i in range(num_layers - 1):
                for ax in axes:
                    ax.axhline(
                        titan.layers[i + 1].radii[0] / 1e3,
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
        # plt.show()
        # plt.savefig(f"./figures/model3_{paper}_{model}.pdf", dpi=300)

# %%
make_models_table(data)

# %%
