#### PROBLEM 1: TRUE-FALSE QUESTIONS
1) The solution of the stochastic integral $\int_0^T \mu \,dW_t$ is $\mu(W_t-W_0)$ and is a random variable itself
2) *The variance of a Wiener process with scale coefficient $\sigma=1$ at time $t$ is $t^2$*: <br>  **False: the variance is $t$.**
3) The standard Drift-Diffusion Model (DDM) assumes that evidence about a dominant alternative accumulates in discrete chunks over time
4) *The first passage time distribution has a closed-form probability density function, but its density can still be evaluated only numerically* (this sounds false, or at least like it should depend on the SDE being studied; for basic 1D brownian motion the FPT dist has a closed-form PDF and is easy to evaluate)
5) *The Euler-Maruyama method can only be used to simulate linear stochastic systems* <br> **False (I think???)**
6) *For any Bayesian analysis, the prior will always have a smaller variance than the posterior* <br> **False: the prior should have larger variance.**
7) In addition to good statistical practices, experimental validation of cognitive models is crucial for ensuring construct validity.
8) *Markov chain Monte Carlo (MCMC) methods approximate a complex posterior distribution through a simpler, yet analytically tractable, distribution* <br> **False: MCMC methods approximate a complex posterior distribution by the empirical distribution of random samples from the true posterior**
9) For most Bayesian problems, the more data we collect, the less influence does the prior exert on the resulting inferences
10) The effective sample size (ESS) estimated from MCMC samplers differs from the total number of samples because the samples are not independent (i.e., exhibit non-zero autocorrelation).

#### Problem 2: DIFFUSION MODEL EXPLORATIONS

As extensively discussed in class, the drift-diffusion model (DDM) generates two response time (RT) distributions, one for each boundary (i.e., lower and upper boundaries). This exercise asks you to first explore a somewhat counterintuitive question about the basic DDM: What differences between the means of the two RT distributions does the the model predict?

To approach this question from a simulation-based perspective, you need to repeatedly solve the forward problem with different parameter configurations and collect the two summary statis- tics, namely, the two empirical means of the resulting RT distributions. First, choose a suitable configuration of the four parameters and vary only the drift rates within a reasonable range (e.g., v ∈ [0.5 − 1.5]) for a total of 25 different drift rates. Make sure that your parameterizations can generate a sufficient number of RTs for both distributions and you don’t end up with the process only reaching the upper boundary. Second, for each of your parameter configurations, generate N = 2000 synthetic observations and estimate the means of the two distributions. What do you observe regarding the mean difference? Describe and interpret your results. (4 points)

In a similar spirit (keeping all parameters fixed and varying one), explore the effects of each of the parameters on the means and standard deviations of the simulated RT distributions, quantify and describe your results. (4 points)



#### PROBLEM 3: PRIOR AND POSTERIOR VARIANCE

$$
\begin{align*}
\mathrm{Var}[\theta|y]&=\mathbb{E}[\theta|y]^2-\mathbb{E}[\theta^2|y]=\left( \int \theta P(\theta|y)\,d\theta \right)^2-\int \theta^2 P(\theta|y)\,d\theta\\
\mathbb{E}[\mathrm{Var}[\theta|y]]&=\int P(y) (\mathbb{E}[\theta|y]^2-\mathbb{E}[\theta^2|y])\,dy\\
&=\int P(y) \left(\int\theta P(\theta|y)d\theta\right)^2\,dy-\iint \theta^2P(y)P(\theta|y) \,d\theta\,dy\\
\\
&=\int P(y) \left(\int\theta P(\theta|y)d\theta\right)^2\,dy-\int \theta^2\left( \int P(y)P(\theta|y) \,dy \right)\,d\theta\\
&=\int P(y) \left(\int\theta P(\theta|y)d\theta\right)^2\,dy-\int \theta^2P(\theta)\,d\theta\\
&=\mathbb{E}[\mathbb{E}[\theta|y]^2]-\mathbb{E}[\theta^2]\\
\end{align*}
$$
$$
\begin{align*}
\mathbb{E}[\theta|y]&=\int \theta P(\theta|y) \, d\theta\\
\mathrm{Var}[\mathbb{E}[\theta|y]]&=\mathbb{E}[\mathbb{E}[\theta|y]]^2-\mathbb{E}[\mathbb{E}[\theta|y]^2]\\
&=\left( \int  P(y)\left( \int \theta P(\theta|y) \, d\theta \right) \,dy\right)^2-\int P(y)\left( \int \theta P(\theta|y) \, d\theta \right)^2\,dy\\
&=\left( \iint  P(y) \theta P(\theta|y) \, d\theta \,dy\right)^2-\int P(y)\left( \int \theta P(\theta|y) \, d\theta \right)^2\,dy\\
&=\left( \int\theta\left( \int  P(y) P(\theta|y) \, dy \right) \,d\theta\right)^2-\int P(y)\left( \int \theta P(\theta|y) \, d\theta \right)^2\,dy\\
&=\left( \int\theta P(\theta) \,d\theta\right)^2-\int P(y)\left( \int \theta P(\theta|y) \, d\theta \right)^2\,dy\\
\\
&=\mathbb{E}[\theta]^2-\mathbb{E}[\mathbb{E}[\theta|y]^2]
\end{align*}
$$
<br>

$$
\begin{align*}
\mathbb{E}[\mathrm{Var}[\theta|y]]+\mathrm{Var}[\mathbb{E}[\theta|y]]&=(\cancel{ \mathbb{E}[\mathbb{E}[\theta|y]^2] }-\mathbb{E}[\theta^2])+(\mathbb{E}[\theta]^2-\cancel{ \mathbb{E}[\mathbb{E}[\theta|y]^2] })\\
&=\mathbb{E}[\theta]^2-\mathbb{E}[\theta^2]\equiv\mathrm{Var}[\theta]\\
&\to \boxed{\mathrm{Var}[\theta]=\mathbb{E}[\mathrm{Var}[\theta|y]]+\mathrm{Var}[\mathbb{E}[\theta|y]]}
\end{align*}
$$
