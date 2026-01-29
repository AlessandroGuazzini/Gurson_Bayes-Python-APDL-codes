import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Use gggrossiTNRW as pyplot style
#plt.style.use("gggrossiTNRW")

from cycler import cycler

# Style
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


# Load database files - if you want to use the full database, uncomment the next line
#database = torch.load("Database_MeshFact_1.pt")

# Load subset database
database = torch.load("Database_MeshFact_1_subset.pt")

# Load displacements
displacements = torch.load("displacements.pt")

# Load fN_vec and epsN_vec
fN_vec = torch.load("fN_vec.pt")
epsN_vec = torch.load("epsN_vec.pt")



# At this point, we have to transform inputs and outputs to prepare them to be fitted by a GPR

# First, let's transform the inputs to a linear scale
# x_fN = torch.log10(fN_vec)
x_fN = fN_vec  # New version
x_epsN = epsN_vec

# Then, calculate the mean and standard deviation of the inputs
mean_fN = torch.mean(x_fN)
mean_epsN = torch.mean(x_epsN)
std_fN = torch.std(x_fN)
std_epsN = torch.std(x_epsN)

# Finally, normalize inputs
x_fN_norm = (x_fN - mean_fN) / std_fN
x_epsN_norm = (x_epsN - mean_epsN) / std_epsN

# Outputs are trickier. The fact that they share a common initial part is detrimental to the GPR.
# We will first import a fundamental curve (no Gurson applied), and then we will subtract it from all the other curves
# This way, the common part will be zero.
# Then, we store the point where the curve starts to deviate from zero, and we will use it to cut the curve,
# resampling it in 100 points, so that the GPR will only see the part where the curve is not zero.

# Importing the fundamental curve
path = os.getcwd()
path = path + "/UTT_Curve_noGurson.txt"
data = np.loadtxt(path, skiprows=1)
force_noGurson = torch.tensor(data[:, 1], dtype=torch.float32)

# Normalize the outputs as planned
tol = 10 # (N) - change it for different \eps_0 (def. 10)
ind = 0  # Trial index
differential_database = {}
transformed_outputs = torch.zeros((len(database), 101))
normalized_inputs = torch.zeros((len(database), 2))
for key in database:
    # Subtracting the fundamental curve from all the curves in Database
    differential_database[key] = database[key] - force_noGurson

    # Now, we have to find the point where the curve starts to deviate from zero

    # Find the first point where the curve is not zero (up to tol)
    first = 0
    for i in range(len(differential_database[key])):
        if np.abs(differential_database[key][i]) > tol:
            first = i
            break

    # Resample the remaining part of the curve in 100 points
    # Recall that the displacements vector is the same for all the curves and is equispaced
    temp = differential_database[key][first:]
    temp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(temp)), temp)

    # Append displacement[first] to the beginning of the vector
    temp = np.append(displacements[first], temp)

    # temp is now a vector of 101 points, which we will use as output for the GPR
    # We also have to normalize the inputs, so we store them in a separate vector
    # normalized_inputs[ind, :] = torch.tensor(
    #     ((np.log10(key[0]) - mean_fN) / std_fN, (key[1] - mean_epsN) / std_epsN)
    # )
    normalized_inputs[ind, :] = torch.tensor(
        ((key[0] - mean_fN) / std_fN, (key[1] - mean_epsN) / std_epsN)
    )  # New version
    transformed_outputs[ind, :] = torch.tensor(temp)
    ind += 1

# We have to normalize the outputs as well, across the whole database
# First, we have to find the mean and standard deviation of the outputs
mean_output = torch.mean(transformed_outputs, dim=0)
std_output = torch.std(transformed_outputs, dim=0)

# Then, we normalize the outputs
normalized_outputs = (transformed_outputs - mean_output) / std_output

# Plot normalized inputs
plt.figure()
plt.scatter(normalized_inputs[:, 0], normalized_inputs[:, 1])
# plt.xlabel("log10(fN)")
plt.xlabel("fN")  # New version
plt.ylabel("epsN")
plt.title("Normalized inputs")
plt.show(block=False)
plt.pause(5)

# Plot normalized outputs
plt.figure()
plt.plot(normalized_outputs.T)
plt.xlabel("Displacement")
plt.ylabel("Normalized force")
plt.title("Normalized outputs")
plt.show(block=False)
plt.pause(5)


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
        output_curve[:first_disp_index] = force_noGurson[:first_disp_index]

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

        output_curve[first_disp_index:] = force_noGurson[
            first_disp_index:
        ] + torch.tensor(interpolated_differences, dtype=torch.float32)

        denormalized_outputs[i, :] = output_curve

    return denormalized_outputs


# Define function that takes a denormalized input and returns the normalized input
def normalize_input(input):
    # return torch.stack(
    #     (
    #         (torch.log10(input[:, 0]) - mean_fN) / std_fN,
    #         (input[:, 1] - mean_epsN) / std_epsN,
    #     ),
    #     dim=1,
    # )
    return torch.stack(
        (
            (input[:, 0] - mean_fN) / std_fN,
            (input[:, 1] - mean_epsN) / std_epsN,
        ),
        dim=1,
    )  # New version


# Define GPR model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=normalized_outputs.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=normalized_outputs.shape[1], rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=normalized_outputs.shape[1]
)
model = MultitaskGPModel(normalized_inputs, normalized_outputs, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# Training loop - 1000 iterations (initially)
training_iterations = 1000
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(normalized_inputs)
    loss = -mll(output, normalized_outputs)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.tensor([[0.0012, 0.25], [0.012, 0.15]]) # Example inputs - can be changed
    test_x = normalize_input(test_x)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Plot predictions
plt.figure()
plt.plot(displacements, denormalize_output(mean[0])[0], label="Prediction 1")
plt.plot(displacements, denormalize_output(mean[1])[0], label="Prediction 2")
plt.plot(displacements, force_noGurson, label="Fundamental curve")
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.title("GPR predictions")
plt.legend()
plt.show()
#plt.show(block=False)
#plt.pause(5)  # serve per aggiornare la finestra


# Save model
torch.save(model.state_dict(), "Multitask_GPR_Model.pt")

# Save likelihood
torch.save(likelihood.state_dict(), "Multitask_GPR_Likelihood.pt")

# Save mean and standard deviation of the outputs
torch.save(mean_output, "Mean_Output.pt")
torch.save(std_output, "Std_Output.pt")

# Save mean and standard deviation of the inputs
torch.save(mean_fN, "Mean_fN.pt")
torch.save(mean_epsN, "Mean_epsN.pt")
torch.save(std_fN, "Std_fN.pt")
torch.save(std_epsN, "Std_epsN.pt")

# Save fundamental curve
torch.save(force_noGurson, "Fundamental_Curve.pt")

# Save displacement
torch.save(displacements, "Displacements.pt")

# Save normalized inputs
torch.save(normalized_inputs, "Normalized_Inputs.pt")

# Save normalized outputs
torch.save(normalized_outputs, "Normalized_Outputs.pt")

# Save tolerance for normalization
torch.save(tol, "Tolerance.pt")


# Print fN_vec and epsN_vec
print("Valori di fN:", fN_vec)
print("Valori di epsN:", epsN_vec)
