import numpy as np


# Number of samples
N=100_000_000


points=np.random.uniform(-1,1,(N,2))
pi_est=4*((points**2).sum(axis=1)<1).mean()


print(pi_est)