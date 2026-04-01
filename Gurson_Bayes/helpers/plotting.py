import matplotlib.pyplot as plt
from cycler import cycler

def apply_plot_style():
    """
    Applies a consistent and premium plotting style using matplotlib rcParams.
    """
    plt.rcParams.update(
        {
            # --- Patch settings ---
            "patch.linewidth": 0.5,
            "patch.facecolor": "#348ABD",  # blue
            "patch.edgecolor": "#EEEEEE",
            "patch.antialiased": True,
            # --- Font settings ---
            "font.size": 15,
            "font.family": "STIXGeneral",
            # 'font.family': 'serif',
            # 'font.serif': ['Times New Roman'],
            # --- Math text ---
            "mathtext.fontset": "stix",
            # --- Axes settings ---
            "axes.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 1,
            "axes.grid": True,
            "axes.titlesize": "x-large",
            "axes.labelsize": "large",
            "axes.labelcolor": "#555555",
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(
                "color",
                [
                    "#E24A33",  # red
                    "#348ABD",  # blue
                    "#988ED5",  # purple
                    "#777777",  # gray
                    "#FBC15E",  # yellow
                    "#8EBA42",  # green
                    "#FFB5B8",  # pink
                ],
            ),
            # --- Tick settings ---
            "xtick.color": "#555555",
            "xtick.direction": "out",
            "ytick.color": "#555555",
            "ytick.direction": "out",
            # --- Grid settings ---
            "grid.color": "#E5E5E5",
            "grid.linestyle": "-",
            # --- Figure settings ---
            "figure.facecolor": "white",
            "figure.edgecolor": "0.50",
            # --- Layout settings ---
            "figure.constrained_layout.use": True,
        }
    )
