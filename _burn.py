# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from burnman import Layer, Planet
from utils import burnman_models, get_temp_info, iterate_layers, models, picks, verify_results

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
    print(paper, model)
    layers: list[Layer] = burnman_models[paper][model]

    temp_info = get_temp_info(paper, model)

    if model == r"Pure water ice":
        temp_info[-1]["top"] = 875

    # temp_info = [
    #     {
    #         "top": 93,
    #         "mode": "adiabatic",
    #     },
    #     {"gradient": 1.2},
    #     {"mode": "adiabatic"},
    #     {"gradient": 0.5},
    # ]

    values = iterate_layers(models[paper][model], temp_info)

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

        # if "mode" in t and t["mode"] == "adiabatic":
        #     options["mode"] = "adiabatic"
        #     options["temperature_top"] = t["top"]
        #     layer.make()
        #     continue

        # if i == 0:
        #     layer.set_temperature_mode("adiabatic", temperature_top=93)
        #     layer.set_pressure_mode(
        #         "self-consistent",
        #         gravity_bottom=values[j, 0, 2],
        #         pressure_top=values[j, -1, 3],
        #     )
        #     layer.make()
        #     print(layer.temperatures)
        #     continue

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
        plt.savefig(f"./figures/model3_{paper}_{model}.pdf", dpi=300)

# %%
