data {
    int<lower=1> N;
    int<lower=1> P;
    matrix[N, P] x;
    array[N] int<lower=0, upper=1> y;
    int<lower=1> N_test;
    matrix[N_test, P] x_test;
}

parameters {
    real intercept;
    vector[P] beta;
}

model {
    intercept ~ normal(0, 1.5);
    beta ~ normal(0, 1.5);

    y ~ bernoulli_logit(intercept + x * beta);
}

generated quantities {
    vector[N_test] p_test;

    for (n in 1:N_test) {
        p_test[n] = inv_logit(intercept + x_test[n] * beta);
    }
}
