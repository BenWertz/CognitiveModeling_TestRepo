data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma2;
}

transformed parameters {
    real<lower=0> sigma;
    sigma = sqrt(sigma2);
}

model {
    //prior
    sigma2 ~ inv_gamma(1, 1);
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    //likelihood
    y ~ normal(alpha + beta * x, sigma);
}

generated quantities {
   array[N] real y_pp;
   for(n in 1:N){
        y_pp[n]=normal_rng(alpha+beta*x[n],sigma);
   }
}