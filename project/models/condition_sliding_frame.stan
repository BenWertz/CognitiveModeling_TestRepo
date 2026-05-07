data {

  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;

  array[N] int<lower=1, upper=K> cond;

  vector[N] response;
  vector[N] target_mu;
  array[N] vector[J] swap_mu;


  array[N, J] int<lower=0, upper=1> swap_in_frame;
  array[N] int<lower=1, upper=J> swap_frame_count;
}

transformed data {
  vector[N] target_cos;
  array[N] vector[J] swap_cos;

  for (n in 1:N) {
    target_cos[n] = cos(response[n] - target_mu[n]);

    for (j in 1:J) {
      swap_cos[n][j] = cos(response[n] - swap_mu[n][j]);
    }
  }
}

parameters {
  vector[K] alpha_m;
  vector[K] alpha_b;

  real<lower=0> kappa_target;
  real<lower=0> kappa_swap;
}

transformed parameters {
  vector<lower=0, upper=1>[K] m_condition;
  vector<lower=0, upper=1>[K] b_condition;

  for (k in 1:K) {
    m_condition[k] = inv_logit(alpha_m[k]);
    b_condition[k] = inv_logit(alpha_b[k]);
  }
}

model {
  real target_norm = -log_modified_bessel_first_kind(0, kappa_target);
  real swap_norm = -log_modified_bessel_first_kind(0, kappa_swap);

  alpha_m ~ normal(0, 1.5);
  alpha_b ~ normal(-0.5, 1.5);
  kappa_target ~ lognormal(log(8), 1);
  kappa_swap ~ lognormal(log(5), 1);

  for (n in 1:N) {
    real mt = m_condition[cond[n]];
    real bt = b_condition[cond[n]];
    real swap_kernel_lse = -1e300;

    real target_lp = log(mt) + target_norm + kappa_target * target_cos[n];

    for (j in 1:J) {
      if (swap_in_frame[n, j] == 1) {
        swap_kernel_lse = log_sum_exp(
          swap_kernel_lse,
          kappa_swap * swap_cos[n][j]
        );
      }
    }

    real swap_lp = log1m(mt) + log(bt) + swap_norm
      + swap_kernel_lse - log(swap_frame_count[n]);
    real guess_lp = log1m(mt) + log1m(bt);

    target += log_sum_exp([target_lp, swap_lp, guess_lp]');
  }
}

generated quantities {
  vector[N] log_lik;

  vector[K] p_target_condition;
  vector[K] p_swap_condition;
  vector[K] p_guess_condition;

  {
    real angle_norm = -log(2 * pi());
    real target_norm = -log_modified_bessel_first_kind(0, kappa_target);
    real swap_norm = -log_modified_bessel_first_kind(0, kappa_swap);

    for (n in 1:N) {
      real mt = m_condition[cond[n]];
      real bt = b_condition[cond[n]];
      real swap_kernel_lse = -1e300;

      real target_lp = log(mt) + target_norm + kappa_target * target_cos[n];

      for (j in 1:J) {
        if (swap_in_frame[n, j] == 1) {
          swap_kernel_lse = log_sum_exp(
            swap_kernel_lse,
            kappa_swap * swap_cos[n][j]
          );
        }
      }

      real swap_lp = log1m(mt) + log(bt) + swap_norm
        + swap_kernel_lse - log(swap_frame_count[n]);
      real guess_lp = log1m(mt) + log1m(bt);

      log_lik[n] = angle_norm + log_sum_exp([target_lp, swap_lp, guess_lp]');
    }
  }

  for (k in 1:K) {
    p_target_condition[k] = m_condition[k];
    p_swap_condition[k] = (1 - m_condition[k]) * b_condition[k];
    p_guess_condition[k] = (1 - m_condition[k]) * (1 - b_condition[k]);
  }
}
