"""
Problem 7 – Multivariate Normal Distribution

This module was developed with the assistance of an LLM (ChatGPT / Claude).
See the bottom of the file for a brief discussion of LLM performance and
verification strategy.
"""

import numpy as np
from scipy.stats import multivariate_normal as scipy_mvn

# ---------------------------------------------------------------------------
# 1.  Stand-alone density function
# ---------------------------------------------------------------------------

def multivariate_normal_density(x, mu, Sigma):
    """
    Compute the probability density of a D-dimensional multivariate normal.

    Parameters
    ----------
    x     : array-like, shape (D,) or (N, D)
            Point(s) at which to evaluate the density.
    mu    : array-like, shape (D,)
            Mean vector.
    Sigma : array-like, shape (D, D)
            Covariance matrix (must be symmetric positive-definite).

    Returns
    -------
    density : float or ndarray of shape (N,)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    D = mu.shape[0]
    diff = x - mu                                  # (D,) or (N, D)
    det_Sigma = np.linalg.det(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)

    norm_const = np.sqrt((2 * np.pi) ** D * det_Sigma)

    # Mahalanobis quadratic form: (x-mu)^T Sigma^{-1} (x-mu)
    if diff.ndim == 1:
        mahal = diff @ inv_Sigma @ diff
    else:  # diff is (N, D)
        mahal = np.sum(diff @ inv_Sigma * diff, axis=1)

    return np.exp(-0.5 * mahal) / norm_const


# ---------------------------------------------------------------------------
# 2.  Comparison with scipy.stats.multivariate_normal
# ---------------------------------------------------------------------------

def compare_with_scipy():
    """Compare multivariate_normal_density against SciPy for three cases."""
    np.random.seed(42)

    test_cases = {}

    # --- Case A: Spherical Gaussian (shared variance, zero covariance) -----
    D = 3
    mu_a = np.zeros(D)
    Sigma_a = 2.0 * np.eye(D)                     # sigma^2 = 2 on every dim
    x_a = np.array([1.0, -0.5, 0.3])
    test_cases["Spherical"] = (x_a, mu_a, Sigma_a)

    # --- Case B: Diagonal Gaussian (different variance per dim) ------------
    mu_b = np.array([1.0, 2.0, 3.0])
    Sigma_b = np.diag([0.5, 2.0, 5.0])
    x_b = np.array([0.8, 2.5, 4.0])
    test_cases["Diagonal"] = (x_b, mu_b, Sigma_b)

    # --- Case C: Full-covariance Gaussian ----------------------------------
    mu_c = np.array([1.0, -1.0])
    # Build a valid positive-definite matrix via A A^T
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    Sigma_c = A @ A.T                              # guaranteed PD
    x_c = np.array([0.5, -0.3])
    test_cases["Full-covariance"] = (x_c, mu_c, Sigma_c)

    print("=" * 65)
    print("Comparison: multivariate_normal_density  vs  scipy")
    print("=" * 65)
    for label, (x, mu, Sigma) in test_cases.items():
        our_val = multivariate_normal_density(x, mu, Sigma)
        sp_val  = scipy_mvn.pdf(x, mean=mu, cov=Sigma)
        match   = np.allclose(our_val, sp_val)
        print(f"\n  [{label}]")
        print(f"    Our density  : {our_val:.12e}")
        print(f"    SciPy density: {sp_val:.12e}")
        print(f"    Match: {match}")
    print()


# ---------------------------------------------------------------------------
# 3.  MultivariateNormal class (scipy-style API)
# ---------------------------------------------------------------------------

class MultivariateNormal:
    """
    Multivariate normal distribution with scipy-like interface.

    Parameters
    ----------
    mu    : array-like, shape (D,)
            The mean vector. 
    Sigma : array-like, shape (D, D)
            The covariance matrix (must be positive-definite)
    """

    def __init__(self, mu, Sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        if len(self.Sigma.shape)!=2:
            raise ValueError(f"Covariance matrix must be 2-dimensional! Shape {self.Sigma.shape} is invalid.")
        if self.Sigma.shape[0]!=self.Sigma.shape[1]:
            raise ValueError(f"Covariance matrix must be square! Shape {self.Sigma.shape} is invalid.")
        if self.Sigma.shape[0]!=self.mu.shape[0]:
            raise ValueError(f"Dimension mismatch between mu (shape={self.mu.shape})) and Sigma (shape={self.Sigma.shape})")

        self.D = self.mu.shape[0]

        # Precompute quantities needed for sampling and log-pdf
        self._inv_Sigma = np.linalg.inv(self.Sigma)
        self._log_norm  = -0.5 * (np.log(2*np.pi) * self.D + np.log(np.linalg.det(self.Sigma)))

        # Cholesky factor for efficient sampling: Sigma = L L^T
        self._L = np.linalg.cholesky(self.Sigma)

    # ---- sampling ---------------------------------------------------------
    def rvs(self, shape=1):
        """
        Draw random variates.

        Parameters
        ----------
        shape : int or tuple of ints
                Output shape.  E.g. shape=5 returns (5, D);
                shape=(3, 4) returns (3, 4, D).

        Returns
        -------
        samples : ndarray of shape (*shape, D)
        """
        if isinstance(shape, int):
            shape = (shape,)
        z = np.random.standard_normal(shape + (self.D,))
        # Transform: x = mu + L @ z  (vectorised over leading dims)
        return self.mu + z @ self._L.T

    # ---- log probability density ------------------------------------------
    def log_pdf(self, x):
        """
        Log probability density at x.

        We prefer the log-density over the raw density because:
        - it avoids underflow or loss of precision for regions with low probability density
        - it combines additively with other log-densities

        Parameters
        ----------
        x : array-like, shape (D,) or (N, D)

        Returns
        -------
        log_density : float or ndarray of shape (N,)
        """
        x = np.asarray(x, dtype=float)
        diff = x - self.mu

        if diff.ndim == 1:
            mahal = diff @ self._inv_Sigma @ diff
        else:
            mahal = np.sum(diff @ self._inv_Sigma * diff, axis=1)

        return self._log_norm - 0.5 * mahal

    # convenience: plain pdf via exp(log_pdf)
    def pdf(self, x):
        """Probability density at x."""
        return np.exp(self.log_pdf(x))


# ---------------------------------------------------------------------------
# 4.  Demonstrate and verify the class
# ---------------------------------------------------------------------------

def verify_class():
    """Verify MultivariateNormal.rvs and .log_pdf against SciPy."""
    np.random.seed(42)

    mu = np.array([1.0, -1.0, 0.5])
    A  = np.array([[2, 0.3, 0.1],
                   [0.3, 1, 0.2],
                   [0.1, 0.2, 1.5]])
    Sigma = A @ A.T  # positive-definite

    dist      = MultivariateNormal(mu, Sigma)
    sp_dist   = scipy_mvn(mean=mu, cov=Sigma)

    # --- log_pdf comparison ------------------------------------------------
    xs = np.random.randn(5, 3)
    our_logpdf = dist.log_pdf(xs)
    sp_logpdf  = sp_dist.logpdf(xs)

    print("=" * 65)
    print("Class verification: log_pdf")
    print("=" * 65)
    for i in range(len(xs)):
        match = np.allclose(our_logpdf[i], sp_logpdf[i])
        print(f"  x[{i}]: ours={our_logpdf[i]:+.8f}  "
              f"scipy={sp_logpdf[i]:+.8f}  match={match}")

    # --- rvs: sample statistics vs analytic moments ------------------------
    samples = dist.rvs(50_000)
    sp_samples = sp_dist.rvs(50_000)
    print(f"\n  Sample mean  (ours)  : {samples.mean(axis=0)}")
    print(f"  Sample mean  (scipy) : {sp_samples.mean(axis=0)}")
    print(f"  True mean            : {mu}")
    print(f"  Sample cov diagonal (ours)  : {np.diag(np.cov(samples.T))}")
    print(f"  Sample cov diagonal (scipy) : {np.diag(np.cov(sp_samples.T))}")
    print(f"  True cov diagonal           : {np.diag(Sigma)}")
    print()


# ---------------------------------------------------------------------------
# 5.  LLM Performance Discussion
# ---------------------------------------------------------------------------

LLM_DISCUSSION = """
LLM PERFORMANCE & VERIFICATION
-------------------------------
The LLM was asked to produce a multivariate normal density function and a
full class with rvs() and log_pdf(). It generated correct, well-structured
code on the first attempt, including:
  - The analytic PDF formula with the normalization constant.
  - Cholesky-based sampling (the standard efficient approach).
  - Vectorized Mahalanobis computation for batches of points.

Minor adjustments were needed for style/formatting and raising exceptions
for invalid inputs, but the mathematical content was correct.

VERIFICATION STRATEGY
  1. Numerical comparison: every density / log-density value was compared
     against scipy.stats.multivariate_normal using np.allclose (relative
     tolerance ~1e-7).  Three parameterizations were tested:
       - Spherical (shared variance, zero covariance)
       - Diagonal  (different variances, zero covariance)
       - Full covariance
  2. Sampling sanity check: 50,000 samples from rvs() were drawn and their
     empirical mean and covariance compared to the analytic parameters.
     Both converge as expected.
  3. log-PDF: log_pdf is preferred over pdf because raw
     densities underflow to 0.0 in high dimensions; log-densities stay
     representable and are additive across i.i.d. observations.
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    compare_with_scipy()
    verify_class()
    print(LLM_DISCUSSION)