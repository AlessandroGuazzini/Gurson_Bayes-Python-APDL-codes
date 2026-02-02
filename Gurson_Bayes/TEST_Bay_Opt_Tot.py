import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy.optimize import minimize, shgo, dual_annealing, direct
from scipy.interpolate import interp1d
from ansys.mapdl.core import launch_mapdl
import warnings
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from cycler import cycler

# Choice of Style
plt.rcParams.update({
    # --- Patch settings ---
    'patch.linewidth': 0.5,
    'patch.facecolor': '#348ABD',  # blue
    'patch.edgecolor': '#EEEEEE',
    'patch.antialiased': True,

    # --- Font settings ---
    'font.size': 15,
    'font.family': 'STIXGeneral',
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],

    # --- Math text ---
    'mathtext.fontset': 'stix',

    # --- Axes settings ---
    'axes.facecolor': 'white',
    'axes.edgecolor': '#555555',
    'axes.linewidth': 1,
    'axes.grid': True,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': '#555555',
    'axes.axisbelow': True,
    'axes.prop_cycle': cycler('color', [
        '#E24A33',  # red
        '#348ABD',  # blue
        '#988ED5',  # purple
        '#777777',  # gray
        '#FBC15E',  # yellow
        '#8EBA42',  # green
        '#FFB5B8'   # pink
    ]),

    # --- Tick settings ---
    'xtick.color': '#555555',
    'xtick.direction': 'out',
    'ytick.color': '#555555',
    'ytick.direction': 'out',

    # --- Grid settings ---
    'grid.color': '#E5E5E5',
    'grid.linestyle': '-',

    # --- Figure settings ---
    'figure.facecolor': 'white',
    'figure.edgecolor': '0.50',
})
plt.rcParams['figure.constrained_layout.use'] = True


# ---PARAMETERS--- #

# Order of the mean used to calculate the loss function
mean_order = 1

# If the loss is greater than this value, interrupt the optimization
# put -0.1 if you want the brute-force solution (default -10)
# loss is negative for ptimization purposes
loss_thresh = -10

# Flag that determines whether to compute the loss from the maximum of the curve onwards or from the whole curve
compute_loss_from_maximum = True

# Beta parameter for Upper Confidence Bound
beta = 2.5

# Flag that constrain pbounds, assuming that the experimental curves are in ascending order of Hydrogen concentration
constrain_pbounds = True

# Number of iterations of Bayesian Optimization (default 10)
num_iterations = 10   # default 10

# Set random seed
np.random.seed(51092)

# Set torch seed
torch.manual_seed(51092)

# ---END OF PARAMETERS--- #


# Define a Logger class that echoes stdout to a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("Bayesian_Optimization_Total.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


sys.stdout = Logger()

# Suppress warnings
warnings.filterwarnings(action="ignore")

# Launch MAPDL instance
path = os.path.join(os.getcwd(), "ANSYS_Folder")
mapdl = launch_mapdl(run_location=path, override=True)

# Import all the curves from the txt files in Experimental_Curves folder in a single dictionary
# Importing the experimental curves
path = os.getcwd()
path = path + "/Experimental_Curves"
files = os.listdir(path)

# Remove file from list if it is not a txt file
for i in range(len(files)):
    if files[i][-3:] != "txt":
        files.remove(files[i])

# Order the files in alphabetical order
files.sort()

# Creating the dictionary
exp_database = {}
data = np.empty((1, 2))
for i in range(len(files)):
    # Importing the txt files
    path = os.getcwd()
    path = path + "/Experimental_Curves/" + files[i]

    # Load file, delimiter is tab
    data = np.loadtxt(path, delimiter=",")

    # Add the data to the dictionary
    exp_database[files[i]] = data.T

# Load tolerance for normalization
tol = torch.load("Tolerance.pt")

# Load normalized inputs and outputs
normalized_inputs = torch.load("Normalized_Inputs.pt")
normalized_outputs = torch.load("Normalized_Outputs.pt")

# Load displacements
displacements = torch.load("Displacements.pt")

# Load force_noGurson
force_noGurson = torch.load("Fundamental_Curve.pt")

# Load fN_vec and epsN_vec
fN_vec = torch.load("fN_vec.pt")
epsN_vec = torch.load("epsN_vec.pt")

# Load mean and standard deviation of inputs
mean_fN = torch.load("Mean_fN.pt")
mean_epsN = torch.load("Mean_epsN.pt")
std_fN = torch.load("Std_fN.pt")
std_epsN = torch.load("Std_epsN.pt")

# Load mean and standard deviation of outputs
mean_output = torch.load("Mean_Output.pt")
std_output = torch.load("Std_Output.pt")

curve_fundamental = torch.stack((displacements, force_noGurson), dim=0)

# Define starting bounds for the acquisition function
pbounds = {
    "fN": (fN_vec[0].item(), fN_vec[-1].item()),
    "epsN": (epsN_vec[0].item(), epsN_vec[-1].item()),
}


# Function that simulates the curve from a given input
def curve_function(fN, epsN):
    mapdl.finish()
    mapdl.clear()

    mapdl.parameters["fN"] = fN
    mapdl.parameters["epsN"] = epsN

    mapdl.input("UTT_Macro.mac")

    # Get force-displacement curve from ANSYS
    # displacement = mapdl.parameters['displacement']
    force = mapdl.parameters["force"]
    force = force[:, 0]

    return torch.tensor(force, dtype=torch.float32)


def loss_function(curve_simulated, curve_to_model):
    # If curve_simulated is a 1D tensor, convert it to a 1xn tensor
    if len(curve_simulated.shape) == 1:
        curve_simulated_2D = torch.unsqueeze(curve_simulated, 0)
    else:
        curve_simulated_2D = curve_simulated.clone().detach()

    loss = np.zeros((curve_simulated_2D.shape[0]))
    for i, sim in enumerate(curve_simulated_2D):
        # Resample curve_simulated on curve_input
        # force_resampled = torch.tensor(
        #     interp1d(
        #         curve_to_model[0, :].detach().numpy(),
        #         displacements.detach().numpy(),
        #         sim.detach().numpy(),
        #     ),
        #     dtype=torch.float32,
        # )
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
            # Get index of maximum of curve_to_model
            max_index = torch.argmax(curve_to_model[1, :]).item()

            # Get residuals by summing the squared differences between the force-displacement curve and the curve_input
            loss[i] = -torch.linalg.norm(
                force_resampled[max_index:] - curve_to_model[1, max_index:],
                dim=None,
                ord=mean_order,
            ) / (curve_to_model.shape[1] ** (1 / mean_order))
        else:
            # Get residuals by summing the squared differences between the force-displacement curve and the curve_input
            loss[i] = -torch.linalg.norm(
                force_resampled - curve_to_model[1, :], dim=None, ord=mean_order
            ) / (curve_to_model.shape[1] ** (1 / mean_order))

    return loss


# Define function that takes a curve and returns the normalized curve
def normalize_curve(curves):
    # If curves is not a tensor, convert it to a tensor
    if not torch.is_tensor(curves):
        curves = torch.tensor(curves, dtype=torch.float32)

    # If curves is a 1D tensor, convert it to a 1xn tensor
    if len(curves.shape) == 1:
        curves_2D = torch.unsqueeze(curves, 0)
    else:
        curves_2D = curves.clone().detach()

    # Initialize normalized_outputs
    normalized_curves = torch.zeros((curves_2D.shape[0], 101))

    # Normalize outputs
    for i, curve in enumerate(curves_2D):
        differential = curve - curve_fundamental[1, :]
        first = 0
        for j in range(len(differential)):
            if np.abs(differential[j]) > tol:
                first = j
                break

        # Resample the remaining part of the curve in 100 points
        # Recall that the displacements vector is the same for all the curves and is equispaced
        temp = differential[first:]
        temp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(temp)), temp)

        # Append displacement[first] to the beginning of the vector
        temp = np.append(displacements[first], temp)

        # temp is now a vector of 101 points, which we will use as output for the GPR
        transformed = torch.tensor(temp, dtype=torch.float32)

        normalized_curves[i, :] = (transformed - mean_output) / std_output

    return normalized_curves


# Define function that takes a normalized output and returns the original curve
def denormalize_output(outputs):
    # If outputs is a 1D tensor, convert it to a 1xn tensor
    if len(outputs.shape) == 1:
        outputs_2D = torch.unsqueeze(outputs, 0)
    else:
        outputs_2D = outputs.clone().detach()

    denormalized_outputs = torch.zeros((outputs_2D.shape[0], len(displacements)))
    for i, out in enumerate(outputs_2D):
        denormalized = out * std_output + mean_output

        # Find element of displacements that is closest to denormalized[0]
        first_disp_index = torch.argmin(
            torch.abs(displacements - denormalized[0])
        ).item()

        output_curve = torch.zeros((len(displacements)))
        output_curve[:first_disp_index] = curve_fundamental[1, :first_disp_index]

        remaining_displacements = displacements[first_disp_index:].detach().numpy()

        interpolated_differences = np.interp(
            remaining_displacements,
            np.linspace(
                remaining_displacements[0],
                remaining_displacements[-1],
                100,
            ),
            denormalized.detach().numpy()[1:],
        )

        output_curve[first_disp_index:] = curve_fundamental[
            1, first_disp_index:
        ] + torch.tensor(interpolated_differences, dtype=torch.float32)

        denormalized_outputs[i, :] = output_curve

    return denormalized_outputs


# Define function that takes a denormalized input and returns the normalized input
def normalize_input(inputs):
    # If inputs is not a tensor, convert it to a tensor
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    # If inputs is a 1D tensor, convert it to a 1x2 tensor
    if len(inputs.shape) == 1:
        inputs_2D = torch.unsqueeze(inputs, 0)
    else:
        inputs_2D = inputs.clone().detach()

    # If any element of the first column is negative, set it to 1e-9
    inputs_2D[inputs_2D[:, 0] < 0, 0] = 1e-9

    res = torch.stack(
        (
            (inputs_2D[..., 0] - mean_fN) / std_fN,
            (inputs_2D[..., 1] - mean_epsN) / std_epsN,
        ),
        dim=1,
    )  # New version

    return res


# Define function that takes a normalized input and returns the denormalized input
def denormalize_input(inputs):
    # If inputs is not a tensor, convert it to a tensor
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    # If inputs is a 1D tensor, convert it to a 1x2 tensor
    if len(inputs.shape) == 1:
        inputs_2D = torch.unsqueeze(inputs, 0)
    else:
        inputs_2D = inputs.clone().detach()

    res = torch.stack(
        (
            inputs_2D[..., 0] * std_fN + mean_fN,
            inputs_2D[..., 1] * std_epsN + mean_epsN,
        ),
        dim=1,
    )  # New version

    # If res is a 1x2 tensor, squeeze it
    if len(res.shape) == 2:
        res = torch.squeeze(res, 0)

    return res


# Define GPR model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=normalized_outputs.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=normalized_outputs.shape[1],
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# Load model and likelihood
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=normalized_outputs.shape[1]
)
model = MultitaskGPModel(normalized_inputs, normalized_outputs, likelihood)

# Load model state dict
model.load_state_dict(torch.load("Multitask_GPR_Model.pt"))

# Load likelihood state dict
likelihood.load_state_dict(torch.load("Multitask_GPR_Likelihood.pt"))

# Set model and likelihood to eval mode
model.eval()
likelihood.eval()


# Now, we define the acquisition function that we want to maximize.
# We use qExpectedImprovement, which is one of the most widely used acquisition functions.
# Since the model is a multitask model, we use a qMultiOutputObjectiveWrapper to specify that we want to maximize the
# objective over all outputs of the model.
def acquisition_function(fN, epsN, model, likelihood, base_samples):
    # Returns acquisition function value, given fN and epsN

    x = normalize_input([fN, epsN])

    with torch.no_grad():
        posterior = likelihood(model(x))

    # Evaluate the distribution of loss_function(x) via MonteCarlo sampling
    samples = posterior.rsample(torch.Size([32]), base_samples=base_samples)
    samples = torch.squeeze(samples)

    loss_samples = loss_function(denormalize_output(samples), curve_to_model)

    # Compute the MonteCarlo estimate of the objective mean
    mc_objective = np.mean(loss_samples, axis=0)

    # Compute the MonteCarlo estimate of the objective variance
    mc_objective_variance = np.var(loss_samples, axis=0)

    # Compute Upper Confidence Bound (UCB)
    ucb = mc_objective + beta * np.sqrt(mc_objective_variance)

    return ucb


min_fN = fN_vec[0].item()
max_epsN = epsN_vec[-1].item()


# ------------------------------------------ START OF THE BIGGEST FOR LOOP ------------------------------------------ #

for exp in exp_database:

   #stop_flag = False  # reset to every new experiment

    # Log date and time
    print("\n" + "Date and time: " + str(np.datetime64("now")))
    print("\n" + "Starting Bayesian Optimization for " + str(exp))

    # Update pbounds, if necessary
    if constrain_pbounds:
        pbounds = {
            "fN": (min_fN, fN_vec[-1].item()),
            "epsN": (epsN_vec[0].item(), max_epsN),
        }
    # Load experimental curve
    curve_to_model = torch.tensor(exp_database[exp], dtype=torch.float32)

    # Convert first row to displacement
    # curve_to_model[0, :] = curve_to_model[0, :] * 10 / 100  # If experimental strain is in percentage
    curve_to_model[0, :] = curve_to_model[0, :] * 10

    # Convert second row to force
    curve_to_model[1, :] = curve_to_model[1, :] * 1.52 * 5.95

    # Evaluate loss_function at the initial points
    training_set_loss = loss_function(
        denormalize_output(normalized_outputs), curve_to_model
    )

    # Get current best
    current_best = np.max(training_set_loss)
    current_best_input = normalized_inputs[np.argmax(training_set_loss)]
    current_curve = torch.squeeze(
        denormalize_output(normalized_outputs[np.argmax(training_set_loss)])
    )

    current_fN = denormalize_input(current_best_input)[0].item()
    current_epsN = denormalize_input(current_best_input)[1].item()

    if constrain_pbounds:
        index = 2
        while current_fN < min_fN or current_epsN > max_epsN:
            print("Current best is out of bounds, trying next best")

            current_index = np.argsort(training_set_loss)[-index]
            current_best = training_set_loss[current_index]
            current_best_input = normalized_inputs[current_index]
            current_curve = torch.squeeze(
                denormalize_output(normalized_outputs[current_index])
            )

            current_fN = denormalize_input(current_best_input)[0].item()
            current_epsN = denormalize_input(current_best_input)[1].item()

            index += 1

    print(
        "Starting Best Parameters: fN = "
        + str(current_fN)
        + ", epsN = "
        + str(current_epsN)
    )
    print("Starting Best Loss (from database): " + str(current_best))

    # Bayesian optimization loop

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iterations):
        # # Optimize acquisition function with Bayesian Optimization
        print(
            "\n"
            + "Iteration "
            + str(i + 1)
            + " of "
            + str(num_iterations)
            + " starting"
        )

        # Generate base samples to be used in the Monte Carlo estimate of the objective,
        # exploiting the reparameterization trick.
        dummy = likelihood(model(torch.tensor([[0.0, 0.0]])))
        base_samples = dummy.get_base_samples(torch.Size([32]))

        # Optimize acquisition function with scipy optimizer
        # res = shgo(lambda x: -acquisition_function(x[0], x[1], model, likelihood, base_samples),
        #            bounds=list(pbounds.values()))
        print("Optimizing Acquisition Function")
        res = shgo(
            lambda x: -acquisition_function(
                x[0], x[1], model, likelihood, base_samples
            ),
            bounds=list(pbounds.values()),
            sampling_method="sobol",
            n=100,
            iters=1,
            minimizer_kwargs={"jac": "3-point"},
        )

        
        new_fN = res.x[0]
        new_epsN = res.x[1]

        # Evaluate the curve function at the new point
        print(
            "New Point being tested: fN = " + str(new_fN) + ", epsN = " + str(new_epsN)
        )
        new_curve = curve_function(new_fN, new_epsN)

    
        new_x = normalize_input([new_fN, new_epsN])

        # Normalize new training curve
        new_y = normalize_curve(new_curve)


        # Evaluate loss_function at the new point
        new_loss = loss_function(new_curve, curve_to_model)[0]


        # Print new loss
        print("New Tested Loss: " + str(new_loss))

        # Get current best et and s0
        current_fN = new_fN if new_loss > current_best else current_fN
        current_epsN = new_epsN if new_loss > current_best else current_epsN
        current_curve = new_curve if new_loss > current_best else current_curve

        # Get current best
        current_best = max(current_best, new_loss)

        # Augment training data with new observations
        normalized_inputs = torch.cat((normalized_inputs, new_x), dim=0)
        normalized_outputs = torch.cat((normalized_outputs, new_y), dim=0)

        # Update model
        model.set_train_data(normalized_inputs, normalized_outputs, strict=False)

        # Print iteration number, current best value and best parameters
        print("Iteration " + str(i + 1) + " of " + str(num_iterations) + " completed")
        print(
            "Current Best Parameters: fN = "
            + str(current_fN)
            + ", epsN = "
            + str(current_epsN)
        )
        print("Current Best Loss: " + str(current_best))

        # Retrain model
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        print("Updating model training")
        training_iterations = 100
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(normalized_inputs)
            loss = -mll(output, normalized_outputs)
            loss.backward()
            optimizer.step()

        # Set back into eval mode
        model.eval()
        likelihood.eval()

        # If loss > threshold, stop optimization
        if current_best > loss_thresh:
            print(f"Loss is sufficient ({current_best:.3f}) -> stop optimization for {exp}")
            # Plot the current best curve vs the curve_to_model, use object-oriented interface
            fig, ax = plt.subplots()
            ax.plot(
                curve_to_model[0, :].detach().numpy(),
                curve_to_model[1, :].detach().numpy(),
                label="Experimental Curve",
            )
            x_exp = curve_to_model[0, :].detach().numpy()
            x_max_exp =x_exp.max()
            x_best = np.concatenate(([0], displacements.detach().numpy()))
            y_best = np.concatenate(([0], current_curve.detach().numpy()))
            mask = x_best <= x_max_exp
            x_best_cut = x_best[mask]
            y_best_cut = y_best[mask]
            ax.plot(
                x_best_cut,
                y_best_cut,
                label="Best Curve",
            )
           
            ax.set_xlabel("Displacement [mm]")
            ax.set_ylabel("Force [N]")
            ax.set_title(f"$C_d = {exp[:-4]}$")
            ax.legend()

            # Write the best parameters on the plot
            ax.text(
                0.7,
                0.35,
                r"$f_N =$ "
                + "{0:.3g}".format(current_fN)
                + "\n"
                + r"$\varepsilon_N =$ "
                + "{0:.3g}".format(current_epsN),
                transform=ax.transAxes,
                verticalalignment="center",
            )

            # Save figure
            save_path = os.path.join(os.getcwd(), exp[:-4] + "_TotalScript.svg")
            save_path2 = os.path.join(os.getcwd(), exp[:-4] + "_TotalScript.pdf")
            fig.savefig(save_path, bbox_inches="tight")
            fig.savefig(save_path2, bbox_inches="tight")
            print(f"Figure saved in {save_path} and in {save_path2}")
          

            # Interrupt the optimization process
            break


    # Print final best value and best parameters
    print(
        "Final Best Parameters: fN = "
        + str(current_fN)
        + ", epsN = "
        + str(current_epsN)
    )
    print("Final Best Loss: " + str(current_best))

    if constrain_pbounds:
        min_fN = current_fN
        max_epsN = current_epsN

    # Plot the current best curve vs the curve_to_model, use object-oriented interface
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
    ax.plot(
        x_best_cut,
        y_best_cut,
        label="Best Curve",
    )
  
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
      #ax.set_title(r"$C_d = $" + exp[:-4]) # added Cd in title r -- +
    ax.set_title(f"$C_d = {exp[:-4]}$")
    ax.legend()

    # Write the best parameters on the plot
    ax.text(
        0.7,
        0.35,
        r"$f_N =$ "
        + "{0:.3g}".format(current_fN)
        + "\n"
        + r"$\varepsilon_N =$ "
        + "{0:.3g}".format(current_epsN)
        + "\n"
        + r"$Loss =$ " + "{0:.3f}".format(abs(current_best)),
        transform=ax.transAxes,
        verticalalignment="center",
    )

    # Save figure (svg)
    #save_path = os.path.join(os.getcwd(), exp[:-4] + "_TotalScript.svg")
    save_path = os.path.join(os.getcwd(), exp[:-4] + "_TotalScript.svg")
    fig.savefig(save_path, bbox_inches="tight")
    #print(f"Figure saved in: {save_path}")
    #fig.savefig(exp[:-4] + "_TotalScript" + ".svg", bbox_inches="tight")

    # Save it also in PDF
    save_path_pdf = os.path.join(os.getcwd(), exp[:-4] + "_TotalScript.pdf")
    fig.savefig(save_path_pdf, bbox_inches="tight", format="pdf")

    # plt.show()
    plt.show(block=False)
    plt.pause(2)  # update window
    #plt.close(fig) # close window



# -------------------------------------------- END OF THE BIGGEST FOR LOOP ------------------------------------------ #

# Print final message
print("\n" + "Bayesian Optimization completed")


mapdl.exit()
