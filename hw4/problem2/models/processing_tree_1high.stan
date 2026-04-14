data {
    // Number of trials
    int<lower=1> N_old;
    // Number of trials
    int<lower=1> N_new;

    // Number of correct identifications of old words in test set
    int<lower=0,upper=N_old> old_ans;
    // Number of misidentified new words in test set
    int<lower=0,upper=N_new> new_ans;
}

parameters {
    // probability of remembering item from old set
    real<lower=0,upper=1> d;
    // probability of guessing unfamiliar item is from old set
    real<lower=0,upper=1> g;
}

transformed parameters {
    // probability of correctly identifying old items
    real p_true_old = d + (1-d)*g;

    // probability of misidentifying new items as old
    real p_false_old = g;
}

model {
    //prior
    target += beta_lpdf(d | 1,1);
    target += beta_lpdf(g | 1,1);

    //likelihood
    //old-item likelihood
    target += binomial_lpmf(old_ans | N_old, p_true_old);
    //new-item likelihood
    target += binomial_lpmf(new_ans | N_new, p_false_old);
}

generated quantities {
    array[2] int pred_ans;
    pred_ans[1] = binomial_rng(N_old, p_true_old);
    pred_ans[2] = binomial_rng(N_new, p_false_old);
}