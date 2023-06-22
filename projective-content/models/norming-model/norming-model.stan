functions {
  // from https://mc-stan.org/docs/2_18/stan-users-guide/truncated-random-number-generation.html
  real normal_lub_rng(real mu, real sigma, real lb, real ub) {
    real p_lb = normal_cdf(lb, mu, sigma);
    real p_ub = normal_cdf(ub, mu, sigma);
    real u = uniform_rng(p_lb, p_ub);
    real y = mu + sigma * Phi(u);
    return y;
  }
}

data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_context;                        // number of contexts
  int<lower=0> N_subj;                           // number of subjects
  int<lower=1,upper=N_context> context[N_resp];  // context corresponding to response n
  int<lower=1,upper=N_subj> subj[N_resp];        // subject corresponding to response n
  vector<lower=0,upper=1>[N_resp] resp;          // bounded slider response     
}

parameters {
  real<lower=0> context_intercept_std;           // the context random intercept standard deviation
  vector[N_context] context_intercept;           // the context random intercepts
  real<lower=0> subj_intercept_std;              // the subject random intercept standard deviation
  vector[N_subj] subj_intercept;                 // the subject random intercepts
  real<lower=0,upper=1> sigma;
}

transformed parameters {
  real mu[N_resp];
  for (n in 1:N_resp)
    mu[n] = inv_logit(context_intercept[context[n]] + subj_intercept[subj[n]]);
}

model {
  context_intercept_std ~ exponential(1);
  subj_intercept_std ~ exponential(1);

  // sample the context intercepts
  context_intercept ~ normal(0, context_intercept_std);

  // sample the subject intercepts
  subj_intercept ~ normal(0, subj_intercept_std);
  
  // sample the responses
  for (n in 1:N_resp)
    resp[n] ~ normal(mu[n], sigma) T[0,1];
}

generated quantities {
  // compute the average context probabilities for the average subject
  vector[N_context] context_prob;

  for (c in 1:N_context) {
    context_prob[c] = inv_logit(
      context_intercept[c]
    );
  }
}