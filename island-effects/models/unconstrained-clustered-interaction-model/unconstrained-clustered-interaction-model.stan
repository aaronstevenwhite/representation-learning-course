data {
  int<lower=2> N_grammaticality_levels;                   // number of grammaticality levels
  int<lower=2> N_interactions;                            // number of interactions to model as discrete
  int<lower=0> N_resp;                                    // number of responses
  int<lower=0> N_subj;                                    // number of subjects
  int<lower=0> N_item;                                    // number of items
  int<lower=2> N_resp_levels;                             // number of possible likert scale acceptability judgment responses
  int<lower=1> N_fixed;                                   // number of fixed predictors
  int<lower=1> N_by_subj;                                 // number of random by-subject predictors
  int<lower=1> N_by_item;                                 // number of random by-item predictors
  matrix[N_resp,N_fixed] fixed_predictors;                // predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_subj] by_subj_predictors;            // by-subject predictors (length and dependency type) including intercept
  matrix[N_resp,N_by_item] by_item_predictors;            // by-item predictors (length and dependency type) including intercept
  int<lower=1,upper=N_interactions> interactions[N_resp]; // interactions to model as discrete 
  int<lower=1,upper=N_subj> subj[N_resp];                 // subject who gave response n
  int<lower=1,upper=N_item> item[N_resp];                 // item corresponding to response n
  int<lower=1,upper=N_resp_levels> resp[N_resp];          // likert scale acceptability judgment responses
}

parameters {
  vector<upper=0>[N_grammaticality_levels-1] penalty;     // grammaticality violation penalty
  simplex[N_grammaticality_levels] gamma;                 // probabilities of grammaticality levels
  vector[N_fixed] fixed_coefs;                            // fixed coefficients (including intercept)
  cov_matrix[N_by_subj] subj_cov;                         // subject random effects covariance
  cov_matrix[N_by_item] item_cov;                         // item random effects covariance              
  vector[N_by_subj] by_subj_coefs[N_subj];                // by-subject coefficients (including intercept)
  vector[N_by_item] by_item_coefs[N_item];                // by-item coefficients (including intercept)
  vector<lower=0>[N_resp_levels-2] jumps;                 // cutpoint distances for each subject
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

  // compute the decrement
  vector[N_grammaticality_levels] decrement;

  for (g in 1:N_grammaticality_levels) {
    if (g == 1) {
      decrement[g] = 0.0;
    } else {
      decrement[g] = decrement[g-1] + penalty[g-1];
    }
  }  

  // compute the acceptability
  real acc[N_resp,N_grammaticality_levels];
  
  for (n in 1:N_resp) {
    for (g in 1:N_grammaticality_levels) {
      acc[n,g] = fixed_predictors[n] * fixed_coefs + 
                by_subj_predictors[n] * by_subj_coefs[subj[n]] + 
                by_item_predictors[n] * by_item_coefs[item[n]] +
                decrement[g];
    }
  }
}

model {
  // sample penalty
  penalty ~ normal(0, 1);

  // initialize by-subject random effects mean to 0
  vector[N_by_subj] subj_mean;
  subj_mean = rep_vector(0.0, N_by_subj);
  
  // initialize by-item random effects mean to 0
  vector[N_by_item] item_mean;
  item_mean = rep_vector(0.0, N_by_item);
  
  // sample the cutpoints distances
  for (j in 1:(N_resp_levels-2))
    jumps[j] ~ gamma(2,1);
  
  // sample the subject intercepts
  for (s in 1:N_subj)
    by_subj_coefs[s] ~ multi_normal(subj_mean, subj_cov);
  
  // sample the item intercepts
  for (i in 1:N_item)
    by_item_coefs[i] ~ multi_normal(item_mean, item_cov);

  // declare log-likelihood of responses corresponding to a particular interaction
  // assuming a particular grammaticality level
  real theta[N_interactions,N_grammaticality_levels];
  
  // initialize log-likelihood of responses corresponding to a particular interaction
  // to the log-prior on the membership probabilities
  for (i in 1:N_interactions) {
    for (g in 1:N_grammaticality_levels) {
      theta[i,g] = log(gamma[g]);
    }
  }
  
  // add the log-likelihood of each response corresponding to a particular interaction
  // assuming a particular grammaticality level 
  for (n in 1:N_resp) {
    for (g in 1:N_grammaticality_levels) {
      theta[interactions[n],g] += ordered_logistic_lpmf(
        resp[n] | acc[n,g], cutpoints
      );
    }
  }
  
  // compute log-likelihood of all responses corresponding to a particular interaction
  // by summing over the likelihood assuming a particular grammaticality level
  for (i in 1:N_interactions) {
    target += log_sum_exp(theta[i]);
  }
}

generated quantities {
  // declare log-likelihood of all responses corresponding to a particular interaction
  // by summing over the likelihood assuming a particular grammaticality level
  real log_lik_grouped[N_interactions];
  
  // compute log-likelihood of all responses corresponding to a particular interaction
  // by summing over the likelihood assuming a particular grammaticality level
  for (i in 1:N_interactions) {
    log_lik_grouped[i] = log_sum_exp(theta[i]);
  }

  // declare probability of particular grammaticality level for each interaction
  real log_membership[N_interactions,N_grammaticality_levels];
  
  // compute probability of particular grammaticality level for each interaction
  // by dividing the likelihood under that level by the sum over likelihoods
  // across levels
  for (i in 1:N_interactions) {
    for (g in 1:N_grammaticality_levels) {
      log_membership[i,g] = theta[i,g] - log_lik_grouped[i];
    }
  }

  // declare the log-likelihood (really, log-pointwise predictive density) of 
  // each data point
  real log_lik[N_resp];
  
  // compute the log-likelihood (really, log-pointwise predictive density) of 
  // each data point by weighting the likelihood assuming a particular 
  // grammaticality level by the membership probability of that level 
  // for the interaction corresponding to the data point.
  for (n in 1:N_resp) {
    real log_lik_by_level[N_grammaticality_levels];
    for (g in 1:N_grammaticality_levels) {
      log_lik_by_level[g] = log_membership[interactions[n],g] + 
                            ordered_logistic_lpmf(resp[n] | acc[n,g], cutpoints);
    }
    log_lik[n] = log_sum_exp(log_lik_by_level);
  }
}