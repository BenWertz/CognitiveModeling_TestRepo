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

#### PROBLEM 4: NORMAL-NORMAL MODEL

Let's say you are (for some reason) extremely interested in how fingernails grow. You could probably find papers studying human fingernail growth rates, but you're bizarrely passionate about this subject so you want to do it yourself. Your prior is that the growth rate is roughly $4.23\pm 1.6$ mm/month, but you can't be sure, so you round up 30 people with similar heights, clip one of their index fingernails, and round them up again 1 month later to measure how much they've grown. Because you're only measuring a few millimeters of growth and it's possible the growth rate probably isn't 100% consistent, you also assign a 1.6 mm/month uncertainty to your own measurements.

I derived the posterior for normal prior and likelihood (I did this in a Desmos graph at https://www.desmos.com/calculator/u7tnirbome). The result is that, for a prior $p(\mu)\sim\mathcal{N}(\mu_0,\sigma_0)$ and a likelihood function $p(y|\mu)\sim\mathcal{N}(\mu,\sigma^2)$ is also a normal distribution:
$$P(\mu|y_i)=\mathcal{N}\left(\tilde{\mu}=\frac{\mu_0/\sigma_0^2+y_i/\sigma^2}{1/\sigma_0^2+1/\sigma^2},\tilde{\sigma}=\sqrt{1/\sigma_0^2+1/\sigma^2}\right)$$

The posterior evalutated for the full sample $y$ is $$P(\mu|y)\propto\prod_{i=1}^NP(\mu|y_i)$$
This is also a normal distribution, but it proved simpler to calculate the (log) probability density for each data point and multiply them, then normalize to get the final posterior PDF.


#### PROBLEM 6: ESTIMATING THE DRIFT-DIFFUSION MODEL

The parameter that best reflects difficulty is the **drift rate** $v$, because it captures the quality and speed of evidence accumulation.

From the estimates, **condition 1** had the lower drift rates on average, so I interpreted **condition 1 as the high-interference (more difficult) field** and Condition 2 as the low-interference / easier field.
