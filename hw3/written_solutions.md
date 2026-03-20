#### PROBLEM 1: TRUE-FALSE QUESTIONS
1) The solution of the stochastic integral $\int_0^T \mu \,dW_t$ is $\mu(W_t-W_0)$ and is a random variable itself
2) *The variance of a Wiener process with scale coefficient $\sigma=1$ at time $t$ is $t^2$*: <br>  **False: the variance is $t$.**
3) The standard Drift-Diffusion Model (DDM) assumes that evidence about a dominant alternative accumulates in discrete chunks over time
**False:** in the standard DDM, evidence builds up continuously over time, not in separate chunks.
4) *The first passage time distribution has a closed-form probability density function, but its density can still be evaluated only numerically* (this sounds false, or at least like it should depend on the SDE being studied; for basic 1D brownian motion the FPT dist has a closed-form PDF and is easy to evaluate)
5) *The Euler-Maruyama method can only be used to simulate linear stochastic systems* <br> **False (I think???)**
6) *For any Bayesian analysis, the prior will always have a smaller variance than the posterior* <br> **False: the prior should have larger variance.**
7) In addition to good statistical practices, experimental validation of cognitive models is crucial for ensuring construct validity.
8) *Markov chain Monte Carlo (MCMC) methods approximate a complex posterior distribution through a simpler, yet analytically tractable, distribution* <br> **False: MCMC methods approximate a complex posterior distribution by the empirical distribution of random samples from the true posterior**
9) For most Bayesian problems, the more data we collect, the less influence does the prior exert on the resulting inferences
10) The effective sample size (ESS) estimated from MCMC samplers differs from the total number of samples because the samples are not independent (i.e., exhibit non-zero autocorrelation).

#### Problem 2: DIFFUSION MODEL EXPLORATIONS

I simulated 2000 trials for each setting and changed one parameter at a time. For drift rate, I used 25 values from $0.5$ to $1.5$. The main thing I saw was that the upper-bound mean RT was always a little bigger than the lower-bound mean RT, but not by much (around $0.02$ to $0.19$ seconds). As $v$ increased, both mean RTs got smaller.

For the other parameters: bigger $a$ made RTs slower and more spread out, bigger $\beta$ made upper responses faster and lower responses slower, and bigger $\tau$ mostly just added the same amount of time to both. So overall it matched the usual DDM interpretation.



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

#### PROBLEM 6: ESTIMATING THE DRIFT-DIFFUSION MODEL

The parameter that best reflects difficulty is the **drift rate** $v$, because it captures the quality and speed of evidence accumulation.

From the estimates, **condition 1** had the lower drift rates on average, so I interpreted **condition 1 as the high-interference (more difficult) field** and Condition 2 as thelow-interference / easier field.
