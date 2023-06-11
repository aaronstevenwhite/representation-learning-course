data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_item;                           // number of items
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1,upper=N_item> item[N_resp];        // item corresponding to response n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses
  int<lower=0,upper=1> marginalize_raneffs;      // whether to marginalize out the random effects in computing the likelihood        
}
parameters {
  real beta;                                     // the grand average
  real<lower=0> item_intercept_var;              // the item random intercept variance
  vector[N_item] item_intercept;                 // the item random intercepts
  ordered[N_resp_levels-1] cutpoints;            // cutpoints
}
transformed parameters {
  real mu[N_resp];
  for (n in 1:N_resp)
    mu[n] = beta + item_intercept[item[n]];
}
model {  
  // sample the item intercepts
  item_intercept ~ normal(0, item_intercept_var);
  
  // sample the cutpoints
  cutpoints ~ normal(0, 1);
  
  // sample the responses
  for (n in 1:N_resp)
    resp[n] ~ ordered_logistic(mu[n], cutpoints);
}

generated quantities {
  real log_lik[N_resp];
  
  if (marginalize_raneffs == 0) {
    for (n in 1:N_resp) {
      log_lik[n] = ordered_logistic_lpmf(resp[n] | mu[n], cutpoints);
    }
  } else {
    real item_intercept_simulated[N_item];
    real mu_simulated[N_item];
    real log_lik_resp_level_by_item[N_resp_levels,N_item];
    real log_lik_resp_level[N_resp_levels];
    
    for (i in 1:N_item) {
      item_intercept_simulated[i] = normal_rng(0, item_intercept_var);
      mu_simulated[i] = beta + item_intercept_simulated[i];
      
      for (l in 1:N_resp_levels) {
        log_lik_resp_level_by_item[l,i] = ordered_logistic_lpmf(l | mu_simulated[i], cutpoints);
      }
    }
    
    for (l in 1:N_resp_levels) {
      log_lik_resp_level[l] = log_sum_exp(log_lik_resp_level_by_item[l]) - log(N_item);
    }
    
    for (n in 1:N_resp) {
      log_lik[n] = log_lik_resp_level[resp[n]];
    }
  }
}