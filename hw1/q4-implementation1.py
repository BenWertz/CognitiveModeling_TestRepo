import numpy as np


def monte_carlo_pi(samples:int = 50000) -> float:
   
    inside:int = 0
    outside:int = 0


    for i in range(samples):


        x:np.float32 = np.random.uniform(-1, 1)
        y:np.float32 = np.random.uniform(-1, 1)


        if ((x**2)+(y**2) > 1):
            outside += 1
        else:
            inside += 1


    total:int = inside + outside
   
    ratio_inside:np.float32 = inside / total


    return 4 * ratio_inside




if(__name__ == "__main__"):
    estimation = monte_carlo_pi(100000)
    print(estimation)