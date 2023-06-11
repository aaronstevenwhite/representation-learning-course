data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_subj;                           // number of subjects
  int<lower=0> N_item;                           // number of items
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1> N_fixed;                          // number of fixed predictors
  int<lower=1> N_by_subj;                        // number of random by-subject predictors
  int<lower=1> N_by_item;                        // number of random by-item predictors
  matrix[N_resp,N_fixed] fixed_predictors;       // predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_subj] by_subj_predictors;   // by-subject predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_item] by_item_predictors;   // by-item predictors (length and dependency type) including intercept
  int<lower=1,upper=N_subj> subj[N_resp];        // subject who gave response n
  int<lower=1,upper=N_item> item[N_resp];        // item corresponding to response n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses
  int<lower=0,upper=1> marginalize_raneffs;      // whether to marginalize out the random effects in computing the likelihood
}

parameters {
  vector[N_fixed] fixed_coefs;                   // fixed coefficients (including intercept)
  cov_matrix[N_by_subj] subj_cov;                // subject random effects covariance
  cov_matrix[N_by_item] item_cov;                // item random effects covariance              
  vector[N_by_subj] by_subj_coefs[N_subj];       // by-subject coefficients (including intercept)
  vector[N_by_item] by_item_coefs[N_item];       // by-item coefficients (including intercept)
  ordered[N_resp_levels-1] cutpoints;            // cutpoints
}

transformed parameters {
  real mu[N_resp];
  for (n in 1:N_resp) {
    mu[n] = fixed_predictors[n] * fixed_coefs + 
            by_subj_predictors[n] * by_subj_coefs[subj[n]] + 
            by_item_predictors[n] * by_item_coefs[item[n]];
  }
}

model {
  // initialize by-subject random effects mean to 0
  vector[N_by_subj] subj_mean;
  subj_mean = rep_vector(0.0, N_by_subj);
  
  // initialize by-item random effects mean to 0
  vector[N_by_item] item_mean;
  item_mean = rep_vector(0.0, N_by_item);
  
  // sample the cutpoints
  for (k in 1:N_resp_levels-1)
    cutpoints[k] ~ normal(0, 1);
  
  // sample the subject intercepts
  for (s in 1:N_subj)
    by_subj_coefs[s] ~ multi_normal(subj_mean, subj_cov);
  
  // sample the item intercepts
  for (i in 1:N_item)
    by_item_coefs[i] ~ multi_normal(item_mean, item_cov);
  
  // sample the responses
  for (n in 1:N_resp) {
    resp[n] ~ ordered_logistic(mu[n], cutpoints);
  }
}

generated quantities {
  real log_lik[N_resp];
  
  if (marginalize_raneffs == 0) {
    for (n in 1:N_resp) {
      log_lik[n] = ordered_logistic_lpmf(resp[n] | mu[n], cutpoints);
    }
  } else {
    vector[N_by_subj] by_subj_coefs_simulated[N_subj];
    
    vector[N_by_subj] subj_mean;
    subj_mean = rep_vector(0.0, N_by_subj);
    
    for (s in 1:N_subj) {
      by_subj_coefs_simulated[s] = multi_normal_rng(subj_mean, subj_cov);
    }
    
    vector[N_by_item] by_item_coefs_simulated[N_item];
    
    vector[N_by_item] item_mean;
    item_mean = rep_vector(0.0, N_by_item);
    
    for (i in 1:N_item) {
      by_item_coefs_simulated[i] = multi_normal_rng(item_mean, item_cov);
    }
    
    matrix[N_subj,N_item] log_lik_by_subj_item[N_resp];
    
    for (n in 1:N_resp) {
      for (s in 1:N_subj) {
        for (i in 1:N_item) {
          real mu_simulated;
          mu_simulated = fixed_predictors[n] * fixed_coefs + 
                         by_subj_predictors[n] * by_subj_coefs_simulated[s] + 
                         by_item_predictors[n] * by_item_coefs_simulated[i];
          log_lik_by_subj_item[n,s,i] = ordered_logistic_lpmf(resp[n] | mu_simulated, cutpoints);
        }
      }
      log_lik[n] = log_sum_exp(log_lik_by_subj_item[n]) - log(N_subj) - log(N_item);
    }
  }
}