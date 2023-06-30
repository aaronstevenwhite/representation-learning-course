data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_item;                           // number of items
  int<lower=0> N_subj;                           // number of subjects
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1> N_fixed;                          // number of fixed predictors
  int<lower=1> N_by_subj;                        // number of random by-subject predictors
  int<lower=1> N_by_item;                        // number of random by-item predictors
  matrix[N_resp,N_fixed] fixed_predictors;       // predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_item] by_item_predictors;   // by-item predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_subj] by_subj_predictors;   // by-subject predictors (length and dependency type) including intercept
  int<lower=1,upper=N_item> item[N_resp];        // item corresponding to response n
  int<lower=1,upper=N_subj> subj[N_resp];        // subject who gave response n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses
}

parameters {
  vector[N_fixed] fixed_coefs;                   // fixed coefficients (including intercept)
  cov_matrix[N_by_item] item_cov;                // item random effects covariance  
  cov_matrix[N_by_subj] subj_cov;                // subject random effects covariance            
  vector[N_by_item] by_item_coefs[N_item];       // by-item coefficients (including intercept)
  vector[N_by_subj] by_subj_coefs[N_subj];       // by-subject coefficients (including intercept)
  vector<lower=0>[N_resp_levels-2] jumps;        // cutpoint distances for each subject
}

transformed parameters {
  // compute the cutpoints by taking a cumulative sum
  vector[N_resp_levels-1] cutpoints;

  for (c in 1:(N_resp_levels-1)) {
    if (c == 1) {
      cutpoints[c] = 0.0;
    } else {
      cutpoints[c] = cutpoints[c-1] + jumps[c-1];
    }
  }

  // compute the acceptability
  real acc[N_resp];

  for (n in 1:N_resp) {
    acc[n] = fixed_predictors[n] * fixed_coefs + 
             by_item_predictors[n] * by_item_coefs[item[n]] + 
             by_subj_predictors[n] * by_subj_coefs[subj[n]];
  }
}

model { 
  // initialize by-item random effects mean to 0
  vector[N_by_item] item_mean = rep_vector(0.0, N_by_item);

  // sample the item coefficients
  for (i in 1:N_item)
    by_item_coefs[i] ~ multi_normal(item_mean, item_cov);

  // sample the cutpoints distances
  for (j in 1:(N_resp_levels-2))
    jumps[j] ~ gamma(2,1);
  
  // initialize by-subject random effects mean to 0
  vector[N_by_subj] subj_mean = rep_vector(0.0, N_by_subj);

  // sample the subject coefficients
  for (s in 1:N_subj)
    by_subj_coefs[s] ~ multi_normal(subj_mean, subj_cov);

  // sample the responses
  for (n in 1:N_resp) {
    resp[n] ~ ordered_logistic(acc[n], cutpoints);
  }
}

generated quantities {
  real log_lik[N_resp];
  
  for (n in 1:N_resp) {
    log_lik[n] = ordered_logistic_lpmf(resp[n] | acc[n], cutpoints);
  }
}