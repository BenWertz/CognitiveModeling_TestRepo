import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

np.random.seed(9982472)

# sample size
N = 200

# true model parameters
with open("true_model_params.json") as f:
    true_params=json.load(f)
alpha_0 = true_params["alpha_0"]
beta_0  = true_params["beta_0"]
sigma_0 = true_params["sigma_0"]

# X-axis sampling
X_mean  = 0.7
X_sigma = 0.8
X = np.random.normal(
    X_mean,
    X_sigma,
    N
)

Y = np.random.normal(
    alpha_0+beta_0*X,
    sigma_0,
    N
)

plt.scatter(
    X, Y,
    marker="+", c="b", s=20, alpha=0.5,
    label="Datapoints"
)

pl_X = np.linspace(-2.5, 3.5, 2)
pl_Y = alpha_0 + beta_0 * pl_X
plt.plot(
    pl_X,
    pl_Y,
    "k--",
    label=f"True parameters:\n$\\alpha$={alpha_0} $\\beta$={beta_0} $\\sigma$={sigma_0}"
)
plt.fill_between(
    pl_X,
    pl_Y - sigma_0,
    pl_Y + sigma_0,
    color="k",
    alpha=0.1,
    label="1$\\sigma$ interval"
)
plt.minorticks_on()
plt.grid(which="major", alpha=0.3)
plt.grid(which="minor", alpha=0.1)
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-2.5,3.5)
plt.ylim(-30,20)
plt.legend(fancybox=False)
plt.savefig(f"figures/generated_data_N{N}.png")

output = pd.DataFrame(
    data={"X":X, "Y":Y}
)
output.to_csv(
    f"data/generated_data_N{N}.csv",
    index=False
)