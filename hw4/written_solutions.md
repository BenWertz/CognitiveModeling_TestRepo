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

## Problem 5

Our current idea for the final project is to develop a hierarchical Bayesian model to analyze a Kaggle dataset on the effects of background noise levels on a subject's ability to focus ([link](https://kaggle.com/datasets/sidramazam/impact-of-background-noise-on-human-focus-dataset)).

The goal would be to predict the `focus_duration_minutes` and `task_completion_quality` fields from some combination of the various other fields, definitely including noise volume and noise type. The goal would be parameter estimation, creating some kind of distribution for the parameters of a model for the effects of background noise on cognitive ability.

There is existing cognitive science research modeling distraction, but honestly I'm not sure what model to use. I'm interested in some kind of SDE accumulator model, but we'll need to do more research to figure this out.

The hierarchical nature of the model could be used to deal with differences between participants not expressed in the data. For example, in some kind of simple regression model where `focus_duration_minutes` depends on `age` and `noise_volume_level` $$\verb|focus_duration_minutes|\sim \mathcal{N}(\alpha\cdot\verb|age|+\beta\cdot \verb|noise_volume_level|+\gamma,\sigma)$$
$$\alpha\sim p(\alpha),\quad \beta\sim p(\beta),\quad \gamma\sim p(\gamma),\quad \sigma\sim p(\sigma)$$
we could introduce a hierarchical element by arguing that $\beta$, which represents something like a person's noise tolerance, could differ between participants: $\beta\sim \mathcal{N}(\beta_0,\sigma_\beta)$ with some hyperpriors $\beta_0\sim p(\beta_0)$, $\sigma_\beta \sim p(\sigma_\beta)$.

Unlike some of the in-class examples, we have enough data (500 participants) to have a training/test split and use information criteria to criticize/validate the model. How exactly this works will depend on the kind of model we pick.

_Note: obviously there is a lot we haven't figured out about this project yet. I think this is a good idea for a project but I'm not sure how to implement it yet and we've both had a lot of other things going on. Sorry we don't have more yet -Ben_