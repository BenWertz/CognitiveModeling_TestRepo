## Problem 1

1. Direct K-fold cross-validation requires K model re-fits, which may be computationally demanding, especially when inverse inference is costly.

2. Bayes factors (BFs) are relative measures, that is, they cannot differentiate between “equally good” and “equally bad” models.

3. **False**: Marginal likelihoods and, by extension, Bayes factors (BFs) cannot be used to compare models with different likelihoods.  
    - Reason: Bayes factors can compare different models as long as they are fit to the same observed data.

4.  **False**: Both the Binomial and the Dirichlet distribution can be formulated as special cases of the Multinomial distribution.
    - Reason: The binomial distribiution is a special case of the multinomial, but the Dirichlet distribution isn't: rather, it's a generalization of the beta distribution and is a conjugate prior to the multinomial distribution.

5. Bayesian leave-one-out cross-validation (LOO-CV) relies on the posterior predictive distribution of left-out data points.

6. **False**: The Akaike Information Criterion (AIC) penalizes model complexity indirectly through the variance of a model’s marginal likelihood.
	- Reason: AIC penalizes model complexity directly, since it has the `2K` term for `K` free parameters.

7. **False**: The log-predictive density (LPD) is a relative metric of model complexity.
	- Reason: LPD measures predictive performance, not model complexity.

8. **False**: The LPD can be approximated by evaluating the likelihood of each posterior draw (e.g., as provided by an MCMC sampler) and taking the average of all resulting likelihood values.
    - Reason: For LPD you average the likelihoods first and then take the log.

9. Bayes factors do not depend on the prior odds, that is, the ratio of prior model probabilities $p(\mathcal{M}_1)/p(\mathcal{M}_1)$.

10. **False:** You should always prefer information criteria to cross-validation in terms of estimation predictive performance.
	- Reason: Neither method is always best. both are just approximations of out-of-sample performance.


## Problem 2

## Problem 3

## Problem 4

## Problem 5
