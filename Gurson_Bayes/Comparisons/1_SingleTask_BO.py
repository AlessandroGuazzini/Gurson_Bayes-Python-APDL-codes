import os
import numpy as np
import torch
import gpytorch
from scipy.optimize import shgo
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
num_iterations = 10  # default 10
beta = 2.5  # Parameter for UCB
n_initial_points = 3  # Points to run before passing to GP

np.random.seed(51092)
torch.manual_seed(51092)
# ---END OF PARAMETERS--- #


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("SingleTask_BO.log", "a")

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
if not os.path.exists(exp_path):
    raise FileNotFoundError(f"Folder not found: {exp_path}")

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

mean_fN = torch.load("Mean_fN.pt")
mean_epsN = torch.load("Mean_epsN.pt")
std_fN = torch.load("Std_fN.pt")
std_epsN = torch.load("Std_epsN.pt")
os.chdir(curr_dir)  # Change back


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


def normalize_input(inputs):
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if len(inputs.shape) == 1:
        inputs_2D = torch.unsqueeze(inputs, 0)
    else:
        inputs_2D = inputs.clone().detach()

    inputs_2D[inputs_2D[:, 0] < 0, 0] = 1e-9
    res = torch.stack(
        (
            (inputs_2D[..., 0] - mean_fN) / std_fN,
            (inputs_2D[..., 1] - mean_epsN) / std_epsN,
        ),
        dim=1,
    )
    return res


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def acquisition_function(fN, epsN, model, likelihood):
    # Standard UCB for Single Task
    x = normalize_input([fN, epsN])
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(x))
        mean = observed_pred.mean.item()
        std = observed_pred.stddev.item()

        # Upper Confidence Bound
        ucb = mean + beta * std

    return ucb


min_fN = fN_vec[0].item()
max_fN = fN_vec[-1].item()
min_epsN = epsN_vec[0].item()
max_epsN = epsN_vec[-1].item()
pbounds = {
    "fN": (min_fN, max_fN),
    "epsN": (min_epsN, max_epsN),
}

for exp in exp_database:
    exp_name = os.path.splitext(exp)[0]
    print("\nDate and time: " + str(np.datetime64("now")))
    print("\nStarting Single Task BO for Cd=" + str(exp_name))

    curve_to_model = torch.tensor(exp_database[exp], dtype=torch.float32)
    curve_to_model[0, :] = curve_to_model[0, :] * 10
    curve_to_model[1, :] = curve_to_model[1, :] * 1.52 * 5.95

    train_fN_list = []
    train_epsN_list = []
    train_loss_list = []
    curve_history = []

    current_best = -np.inf
    current_fN = None
    current_epsN = None
    current_curve = None

    print(
        f"Generating {n_initial_points} initial random points to start Single Task GP."
    )
    for init_i in range(n_initial_points):
        init_fN = np.random.uniform(min_fN, max_fN)
        init_epsN = np.random.uniform(min_epsN, max_epsN)

        c_curve = curve_function(init_fN, init_epsN)
        c_loss = loss_function(c_curve, curve_to_model)[0]

        train_fN_list.append(init_fN)
        train_epsN_list.append(init_epsN)
        train_loss_list.append(c_loss)
        curve_history.append(c_curve)

        if c_loss > current_best:
            current_best = c_loss
            current_fN = init_fN
            current_epsN = init_epsN
            current_curve = c_curve

        print(f"Init point {init_i+1} tested: loss={c_loss}")

    # GP training and inference loops
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    for i in range(num_iterations):
        print(f"\nIteration {i + 1} of {num_iterations} BO starting")

        # Prepare normalized inputs/outputs
        train_x = normalize_input(np.array(list(zip(train_fN_list, train_epsN_list))))

        train_y_tensor = torch.tensor(train_loss_list, dtype=torch.float32)
        # Normalize training loss for stability
        mean_y = train_y_tensor.mean()
        std_y = train_y_tensor.std()
        if std_y == 0:
            std_y = 1e-6
        train_y_norm = (train_y_tensor - mean_y) / std_y

        model = ExactGPModel(train_x, train_y_norm, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 100
        for _ in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y_norm)
            loss.backward()
            optimizer.step()

        # Optimize Acquisition Function
        print("Optimizing Acquisition Function")

        def ucb_wrapper(x):
            # Evaluate UCB. Shgo MINIMIZES by default, so we return -UCB
            val = acquisition_function(x[0], x[1], model, likelihood)
            return -val

        res = shgo(
            ucb_wrapper,
            bounds=list(pbounds.values()),
            sampling_method="sobol",
            n=100,
            iters=1,
            minimizer_kwargs={"jac": "3-point"},
        )

        new_fN = res.x[0]
        new_epsN = res.x[1]

        print(f"New Point being tested: fN = {new_fN}, epsN = {new_epsN}")

        new_curve = curve_function(new_fN, new_epsN)
        new_loss = loss_function(new_curve, curve_to_model)[0]

        print(f"New Tested Loss: {new_loss}")

        train_fN_list.append(new_fN)
        train_epsN_list.append(new_epsN)
        train_loss_list.append(new_loss)
        curve_history.append(new_curve)

        if new_loss > current_best:
            current_best = new_loss
            current_fN = new_fN
            current_epsN = new_epsN
            current_curve = new_curve

        print(f"Iteration {i + 1} of {num_iterations} completed")
        print(f"Current Best Parameters: fN = {current_fN}, epsN = {current_epsN}")
        print(f"Current Best Loss: {current_best}")

        if current_best > loss_thresh:
            print(
                f"Loss is sufficient ({current_best:.3f}) -> stop optimization for {exp}"
            )
            break

    print(f"Final Best Parameters: fN = {current_fN}, epsN = {current_epsN}")
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

    save_path = os.path.join(os.getcwd(), f"{exp[:-4]}_SingleTask_BO.svg")
    save_path_pdf = os.path.join(os.getcwd(), f"{exp[:-4]}_SingleTask_BO.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path_pdf, bbox_inches="tight", format="pdf")

mapdl.exit()
