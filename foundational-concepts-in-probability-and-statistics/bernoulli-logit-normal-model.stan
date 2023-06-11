data {
    int N;                             // number of datapoints
    real mu;                           // prior mean
    real sigma;                        // prior standard deviation
    int<lower=0, upper=1> x[N];        // datapoints 
}

parameters {
    real logodds;                      // log-odds of success
}

transformed parameters {
    real pi = inv_logit(logodds);      // probability of success
}

model {
    logodds ~ normal(mu, sigma);
    x ~ bernoulli(pi);
}