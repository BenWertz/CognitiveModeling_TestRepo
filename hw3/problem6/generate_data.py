import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(998247)

# sample size
N = 50

# true model parameters
alpha_0 = -11.4
beta_0  = 9.8
sigma_0 = 2.0

# X-axis sampling
X_mean  = 0.7
X_sigma = 0.8
X=np.random.normal(
    X_mean,
    X_sigma,
    N
)

Y=np.random.normal(
    alpha_0+beta_0*X,
    sigma_0,
    N
)

plt.scatter(
    X,Y,
    marker="+",c="b",s=20,alpha=0.5,
    label="Datapoints"
)
plt.plot(
    pl_X:=np.linspace(X.min(),X.max(),2),
    pl_Y:=alpha_0+beta_0*pl_X,
    "k--",
    label=f"True parameters:\n$\\alpha$={alpha_0} $\\beta$={beta_0} $\\sigma$={sigma_0}"
)
plt.fill_between(
    pl_X,
    pl_Y-sigma_0,
    pl_Y+sigma_0,
    color="k",
    alpha=0.1,
    label="1$\\sigma$ interval"
)
plt.minorticks_on()
plt.grid(which="major",alpha=0.3)
plt.grid(which="minor",alpha=0.1)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fancybox=False)
plt.savefig("fig1.png")

output=pd.DataFrame(
    data={"X":X,"Y":Y}
)
output.to_csv("data/generated_data.csv",index=False)