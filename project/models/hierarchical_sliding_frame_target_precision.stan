data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> K;
  int<lower=1> J;

  array[N] int<lower=1, upper=S> subj;
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
  vector[K] target_avg;
  vector[K] swap_avg;

  vector<lower=0>[K] target_sd;
  vector<lower=0>[K] swap_sd;

  matrix[S, K] subj_target_offset;
  matrix[S, K] subj_swap_offset;

  vector[K] log_target_precision_avg;
  vector<lower=0>[K] log_target_precision_sd;
  matrix[S, K] z_target_precision;

  vector[K] log_swap_precision;
}

transformed parameters {
  matrix[S, K] log_target_precision;
  matrix<lower=0>[S, K] target_precision;
  vector<lower=0>[K] swap_precision;

  for (s in 1:S) {
    for (k in 1:K) {
      log_target_precision[s, k] = log_target_precision_avg[k]
        + log_target_precision_sd[k] * z_target_precision[s, k];
      target_precision[s, k] = exp(log_target_precision[s, k]);
    }
  }

  for (k in 1:K) {
    swap_precision[k] = exp(log_swap_precision[k]);
  }
}

model {
  target_avg ~ normal(0, 1.5);
  swap_avg ~ normal(-0.5, 1.5);

  target_sd ~ normal(0, 1);
  swap_sd ~ normal(0, 1);

  to_vector(subj_target_offset) ~ normal(0, 1);
  to_vector(subj_swap_offset) ~ normal(0, 1);

  log_target_precision_avg ~ normal(log(8), 0.75);
  log_target_precision_sd ~ normal(0, 0.5);
  to_vector(z_target_precision) ~ normal(0, 1);

  log_swap_precision ~ normal(log(5), 0.75);

  {
    matrix[S, K] target_norm;
    vector[K] swap_norm;

    matrix[S, K] log_p_target;
    matrix[S, K] log1m_p_target;
    matrix[S, K] log_p_swap;
    matrix[S, K] log1m_p_swap;

    for (s in 1:S) {
      for (k in 1:K) {
        real lt = target_avg[k] + target_sd[k] * subj_target_offset[s, k];
        real ls = swap_avg[k] + swap_sd[k] * subj_swap_offset[s, k];

        log_p_target[s, k] = log_inv_logit(lt);
        log1m_p_target[s, k] = log1m_inv_logit(lt);
        log_p_swap[s, k] = log_inv_logit(ls);
        log1m_p_swap[s, k] = log1m_inv_logit(ls);

        target_norm[s, k] = -log_modified_bessel_first_kind(
          0, target_precision[s, k]
        );
      }
    }

    for (k in 1:K) {
      swap_norm[k] = -log_modified_bessel_first_kind(0, swap_precision[k]);
    }

    for (n in 1:N) {
      int s = subj[n];
      int k = cond[n];
      real swap_kernel_lse = -1e300;

      real target_lp = log_p_target[s, k]
        + target_norm[s, k]
        + target_precision[s, k] * target_cos[n];

      for (j in 1:J) {
        if (swap_in_frame[n, j] == 1) {
          swap_kernel_lse = log_sum_exp(
            swap_kernel_lse,
            swap_precision[k] * swap_cos[n][j]
          );
        }
      }

      real swap_lp = log1m_p_target[s, k]
        + log_p_swap[s, k]
        + swap_norm[k]
        + swap_kernel_lse
        - log(swap_frame_count[n]);

      real guess_lp = log1m_p_target[s, k] + log1m_p_swap[s, k];

      target += log_sum_exp([target_lp, swap_lp, guess_lp]');
    }
  }
}

generated quantities {
  vector[N] log_lik;

  vector[K] p_target_condition;
  vector[K] p_swap_condition;
  vector[K] p_guess_condition;

  vector[K] p_target_typical;
  vector[K] p_swap_typical;
  vector[K] p_guess_typical;

  vector[K] target_precision_condition;
  vector[K] target_precision_typical;

  {
    matrix[S, K] p_target_by_subj;
    matrix[S, K] p_swap_given_not_target_by_subj;
    matrix[S, K] target_norm;
    vector[K] swap_norm;
    real angle_norm;

    angle_norm = -log(2 * pi());

    for (s in 1:S) {
      for (k in 1:K) {
        p_target_by_subj[s, k] = inv_logit(
          target_avg[k] + target_sd[k] * subj_target_offset[s, k]
        );
        p_swap_given_not_target_by_subj[s, k] = inv_logit(
          swap_avg[k] + swap_sd[k] * subj_swap_offset[s, k]
        );
        target_norm[s, k] = -log_modified_bessel_first_kind(
          0, target_precision[s, k]
        );
      }
    }

    for (k in 1:K) {
      swap_norm[k] = -log_modified_bessel_first_kind(0, swap_precision[k]);
    }

    for (n in 1:N) {
      int s = subj[n];
      int k = cond[n];
      real p_target = p_target_by_subj[s, k];
      real p_swap_given_not_target = p_swap_given_not_target_by_subj[s, k];
      real swap_kernel_lse = -1e300;

      real target_lp = log(p_target)
        + target_norm[s, k]
        + target_precision[s, k] * target_cos[n];

      for (j in 1:J) {
        if (swap_in_frame[n, j] == 1) {
          swap_kernel_lse = log_sum_exp(
            swap_kernel_lse,
            swap_precision[k] * swap_cos[n][j]
          );
        }
      }

      real swap_lp = log1m(p_target)
        + log(p_swap_given_not_target)
        + swap_norm[k]
        + swap_kernel_lse
        - log(swap_frame_count[n]);

      real guess_lp = log1m(p_target) + log1m(p_swap_given_not_target);

      log_lik[n] = angle_norm + log_sum_exp([target_lp, swap_lp, guess_lp]');
    }
  }

  for (k in 1:K) {
    real target_sum = 0;
    real swap_sum = 0;
    real guess_sum = 0;
    real target_precision_sum = 0;
    real typical_target = inv_logit(target_avg[k]);
    real typical_swap_given_not_target = inv_logit(swap_avg[k]);

    for (s in 1:S) {
      real p_target = inv_logit(
        target_avg[k] + target_sd[k] * subj_target_offset[s, k]
      );
      real p_swap_given_not_target = inv_logit(
        swap_avg[k] + swap_sd[k] * subj_swap_offset[s, k]
      );

      target_sum += p_target;
      swap_sum += (1 - p_target) * p_swap_given_not_target;
      guess_sum += (1 - p_target) * (1 - p_swap_given_not_target);
      target_precision_sum += target_precision[s, k];
    }

    p_target_condition[k] = target_sum / S;
    p_swap_condition[k] = swap_sum / S;
    p_guess_condition[k] = guess_sum / S;

    p_target_typical[k] = typical_target;
    p_swap_typical[k] = (1 - typical_target) * typical_swap_given_not_target;
    p_guess_typical[k] = (1 - typical_target) * (1 - typical_swap_given_not_target);

    target_precision_condition[k] = target_precision_sum / S;
    target_precision_typical[k] = exp(log_target_precision_avg[k]);
  }
}
