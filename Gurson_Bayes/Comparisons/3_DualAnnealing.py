import os
import numpy as np
import torch
from scipy.optimize import dual_annealing
from scipy.interpolate import interp1d
from ansys.mapdl.core import launch_mapdl
import warnings
import sys
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from helpers.plotting import apply_plot_style

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Apply plotting style
apply_plot_style()

# ---PARAMETERS--- #
mean_order = 1
loss_thresh = -10
compute_loss_from_maximum = True
max_fun_evals = 10  # Limit object evaluations for fair comparison with BO

np.random.seed(51092)
torch.manual_seed(51092)
# ---END OF PARAMETERS--- #


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("DualAnnealing.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()
warnings.filterwarnings(action="ignore")

# Setup paths
ansys_path = os.path.join(base_dir, "..", "ANSYS_Folder")
ansys_path = os.path.abspath(ansys_path)
mapdl = launch_mapdl(run_location=ansys_path, override=True)

exp_path = os.path.join(base_dir, "Experimental_Curves")
# Check if the directory exists
if not os.path.exists(exp_path):
    raise FileNotFoundError(f"Folder not found: {exp_path}")

# List only .txt files
files = [
    f
    for f in os.listdir(exp_path)
    if os.path.isfile(os.path.join(exp_path, f)) and f.endswith(".txt")
]
files.sort()

# Creating the dictionary
exp_database = {}
for f in files:
    path = os.path.join(exp_path, f)
    data = np.loadtxt(path, delimiter=",")
    exp_database[f] = data.T

# Load basic variables for the loss
curr_dir = os.getcwd()
os.chdir(base_dir)  # Change dir temp to load .pt files easily

displacements = torch.load("Displacements.pt")
fN_vec = torch.load("fN_vec.pt")
epsN_vec = torch.load("epsN_vec.pt")
os.chdir(curr_dir)  # Change back

# Keep track of evaluations to reconstruct best curve
evaluation_history = {}


def curve_function(fN, epsN):
    mapdl.finish()
    mapdl.clear()
    mapdl.parameters["fN"] = fN
    mapdl.parameters["epsN"] = epsN
    mac_path = os.path.join(base_dir, "UTT_Macro.mac")
    mapdl.input(mac_path)

    force = mapdl.parameters["force"]
    force = force[:, 0]
    return torch.tensor(force, dtype=torch.float32)


def loss_function(curve_simulated, curve_to_model):
    if len(curve_simulated.shape) == 1:
        curve_simulated_2D = torch.unsqueeze(curve_simulated, 0)
    else:
        curve_simulated_2D = curve_simulated.clone().detach()

    loss = np.zeros((curve_simulated_2D.shape[0]))
    for i, sim in enumerate(curve_simulated_2D):
        interpolator = interp1d(
            displacements.detach().numpy(),
            sim.detach().numpy(),
            kind="linear",
            fill_value="extrapolate",
        )
        force_resampled = torch.tensor(
            interpolator(curve_to_model[0, :].detach().numpy()), dtype=torch.float32
        )
        if compute_loss_from_maximum:
            max_index = torch.argmax(curve_to_model[1, :]).item()
            loss[i] = -torch.linalg.norm(
                force_resampled[max_index:] - curve_to_model[1, max_index:],
                dim=None,
                ord=mean_order,
            ) / (curve_to_model.shape[1] ** (1 / mean_order))
        else:
            loss[i] = -torch.linalg.norm(
                force_resampled - curve_to_model[1, :], dim=None, ord=mean_order
            ) / (curve_to_model.shape[1] ** (1 / mean_order))
    return loss


def objective(x, curve_to_model):
    fN, epsN = x
    print(f"Dual Annealing testing: fN={fN:.5f}, epsN={epsN:.5f}")

    curve = curve_function(fN, epsN)
    loss = loss_function(curve, curve_to_model)[0]  # returns negative loss

    evaluation_history[(fN, epsN)] = (curve, loss)
    # The scipy minimize wants to *minimize* the objective function,
    # so we return -loss, which is positive. The closer to 0, the better.
    print(f"Objective Evaluated: {-loss:.3f}")
    return -loss


def stop_cb(x, f, context):
    # Stop earlier if we fall under the loss_thresh?
    # Return True to stop.
    # Note: f is positive here, so we compare -f > loss_thresh
    current_loss = -f
    if current_loss > loss_thresh:
        print(f"Optimization stopped: reached sufficient loss {current_loss:.3f}")
        return True
    return False


min_fN = fN_vec[0].item()
max_fN = fN_vec[-1].item()
min_epsN = epsN_vec[0].item()
max_epsN = epsN_vec[-1].item()
bounds = [(min_fN, max_fN), (min_epsN, max_epsN)]

for exp in exp_database:
    exp_name = os.path.splitext(exp)[0]
    print("\nDate and time: " + str(np.datetime64("now")))
    print("\nStarting Dual Annealing Optimization for Cd=" + str(exp_name))

    curve_to_model = torch.tensor(exp_database[exp], dtype=torch.float32)
    curve_to_model[0, :] = curve_to_model[0, :] * 10
    curve_to_model[1, :] = curve_to_model[1, :] * 1.52 * 5.95

    evaluation_history.clear()

    # Callback stopping mechanism requires maxfun bound to prevent long runs
    res = dual_annealing(
        objective,
        bounds,
        args=(curve_to_model,),
        maxfun=max_fun_evals,
        callback=stop_cb,
    )

    best_fN, best_epsN = res.x
    # Due to floating point precision during dict lookup, we extract from history by distance
    best_item = None
    best_dist = np.inf
    for k, v in evaluation_history.items():
        dist = (k[0] - best_fN) ** 2 + (k[1] - best_epsN) ** 2
        if dist < best_dist:
            best_dist = dist
            best_item = v

    current_curve, current_best = best_item
    current_fN = best_fN
    current_epsN = best_epsN

    print(f"\nFinal Best Parameters: fN = {current_fN}, epsN = {current_epsN}")
    print(f"Final Best Loss: {current_best}")

    fig, ax = plt.subplots()
    x_exp = curve_to_model[0, :].detach().numpy()
    x_max_exp = x_exp.max()
    x_best = np.concatenate(([0], displacements.detach().numpy()))
    y_best = np.concatenate(([0], current_curve.detach().numpy()))
    mask = x_best <= x_max_exp
    x_best_cut = x_best[mask]
    y_best_cut = y_best[mask]

    ax.plot(
        curve_to_model[0, :].detach().numpy(),
        curve_to_model[1, :].detach().numpy(),
        label="Experimental Curve",
    )
    ax.plot(x_best_cut, y_best_cut, label="Best Curve")

    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    ax.set_title(f"$C_d = {exp[:-4]}$")
    ax.legend()

    ax.text(
        0.7,
        0.35,
        r"$f_N =$ {:.3g}".format(current_fN)
        + "\n"
        + r"$\varepsilon_N =$ {:.3g}".format(current_epsN)
        + "\n"
        + r"$Loss =$ {:.3f}".format(abs(current_best)),
        transform=ax.transAxes,
        verticalalignment="center",
    )

    save_path = os.path.join(os.getcwd(), f"{exp[:-4]}_DualAnnealing.svg")
    save_path_pdf = os.path.join(os.getcwd(), f"{exp[:-4]}_DualAnnealing.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path_pdf, bbox_inches="tight", format="pdf")

mapdl.exit()
