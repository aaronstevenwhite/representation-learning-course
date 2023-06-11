data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses 
}

parameters {
  real mu;                                       // the grand average
  ordered[N_resp_levels-1] cutpoints;            // cutpoints
}

model {  
  // sample the cutpoints
  cutpoints ~ normal(0, 1);
  
  // sample the responses
  for (n in 1:N_resp)
    resp[n] ~ ordered_logistic(mu, cutpoints);
}

generated quantities {
  real log_lik[N_resp];
  
  for (n in 1:N_resp)
    log_lik[n] = ordered_logistic_lpmf(resp[n] | mu, cutpoints);
}