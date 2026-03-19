import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# true parameters for generating data
mu_actual=3.6
sigma2_actual=0.2
N=10
Y=np.random.normal(mu_actual,np.sqrt(sigma2_actual),N)

# prior hyperparameters
mu_0=4.23
sigma_0=1.6
prior_model=stats.Normal(mu=mu_0,sigma=sigma_0)

# known data variance from likelihood fn
sigma2_lk=0.2

# conjugate posterior dist
mu_post=(mu_0/sigma_0**2+Y/sigma2_lk)/(1/sigma_0**2+1/sigma2_lk)
sigma_post=1/np.sqrt(1/sigma_0**2+1/sigma2_lk)
# posterior_model=stats.Normal(mu=mu_post,sigma=sigma_post)

N_plot=1000
X=np.linspace(1,10,N_plot)
Y_prior_pdf=prior_model.pdf(X)
Y_posterior_lpdf=np.zeros_like(X)
for i in range(len(Y)):
    Y_posterior_lpdf+=stats.Normal(mu=mu_post[i],sigma=sigma_post).logpdf(X)
Y_posterior_pdf=np.exp(Y_posterior_lpdf)
Y_posterior_pdf/=np.trapz(Y_posterior_pdf,X)

plt.scatter(Y,np.zeros_like(Y),marker="x")

plt.plot(X,Y_prior_pdf,"b-",label="prior")
plt.plot(X,Y_posterior_pdf,"r-",label="posterior")

plt.axvline(mu_actual,color="b",ls="--")

plt.show()