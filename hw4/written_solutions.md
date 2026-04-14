## Problem 1

1. Direct K-fold cross-validation requires K model re-fits, which may be computationally demanding, especially when inverse inference is costly.

2. Bayes factors (BFs) are relative measures, that is, they cannot differentiate between “equally good” and “equally bad” models.

3. Marginal likelihoods and, by extension, Bayes factors (BFs) cannot be used to compare models with different likelihoods.

4. Both the Binomial and the Dirichlet distribution can be formulated as special cases of the Multinomial distribution.
	- **False**: The Dirichlet distribution is a generalization of the beta distribution and is conjugate to the multinomial distribution, not a special case of the multinomial distribution.

5. Bayesian leave-one-out cross-validation (LOO-CV) relies on the posterior predictive distribution of left-out data points.
	- *not sure?*

6. The Akaike Information Criterion (AIC) penalizes model complexity indirectly through the variance of a model’s marginal likelihood.
	- **False**: AIC penalizes model complexity directly through the number of free model parameters.

7. The log-predictive density (LPD) is a relative metric of model complexity.
	- **False?**: LPD is a relative metric, but it's a metric of predictive performance, and doesn't seem to have anything to do with the complexity of the model.

8. The LPD can be approximated by evaluating the likelihood of each posterior draw (e.g., as provided by an MCMC sampler) and taking the average of all resulting likelihood values.
	- That seems right: according to slides, $$\begin{align*}
\mathrm{LPD}&=\log\left[ \int p(\mathbf{y}_{new}|\mathbf{\theta},\mathbf{x_{new}})p(\theta|\mathcal{D}) \, d\theta \right]\\
&\approx\log\left[ \frac{1}{S}\sum_{s=1}^{S}p(\mathbf{y_{new}}|\theta_{s},\mathbf{x_{new}})\right]\quad\text{for}\quad \theta_{s}\sim p(\theta|\mathcal{D})\\
\end{align*}$$ which is just the log of the averaged likelihood of each posterior draw.

9. Bayes factors do not depend on the prior odds, that is, the ratio of prior model probabilities $p(\mathcal{M}_1)/p(\mathcal{M}_1)$.

    - **False**: Bayes factors are proportional to posterior odds by the inverse of the prior odds. $$\mathrm{BF_{12}}=\frac{p(\mathbf{y}|\mathcal{M}_{1})}{p(\mathbf{y}|\mathcal{M}_{2})}=\frac{p(\mathcal{M}_{1}|\mathbf{y})}{p(\mathcal{M}_{2}|\mathbf{y})}\left(  \frac{p(\mathcal{M}_{1})}{p(\mathcal{M}_{2})} \right)^{-1}$$
10. You should always prefer information criteria to cross-validation in terms of estimation predictive performance.
	- *not sure. sounds false?*

## Problem 2

## Problem 3

## Problem 4

## Problem 5
