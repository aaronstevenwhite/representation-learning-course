data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_subj;                           // number of subjects
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1,upper=N_subj> subj[N_resp];        // subject who gave response n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses 
}

parameters {
  real acc_mean;                                 // the mean acceptability
  real<lower=0> subj_std;                        // the standard deviation of the first cutpoint
  real<lower=0> subj_alpha;                      // the alpha parameter for the jump distribution
  real<lower=0> subj_beta;                       // the beta parameter for the jump distribution
  ordered[N_resp_levels-1] cutpoints[N_subj];    // cutpoints for each subject
}

transformed parameters {
  // compute the jumps by taking a cumulative difference
  matrix[N_subj,N_resp_levels-2] jumps;

  for (s in 1:N_subj) {
    for (c in 2:(N_resp_levels-1)) {
      jumps[s,c-1] = cutpoints[s,c] - cutpoints[s,c-1];
    }
  }
}

model {  
  // sample the first cutpoints
  cutpoints[,1] ~ normal(0, subj_std);

  // sample the cutpoints distances
  for (j in 1:(N_resp_levels-2))
    jumps[,j] ~ gamma(subj_alpha,subj_beta);
  
  // sample the responses
  for (n in 1:N_resp)
    resp[n] ~ ordered_logistic(
      acc_mean, cutpoints[subj[n]]
    );
}

generated quantities {
  // compute the log-likelihood
  real log_lik[N_resp];
  
  for (n in 1:N_resp)
    log_lik[n] = ordered_logistic_lpmf(
      resp[n] | acc_mean, cutpoints[subj[n]]
    );

  // compute the mean for each cutpoint across subjects
  vector[N_resp_levels-1] cutpoints_mean; 

  for (c in 1:(N_resp_levels-1))
    cutpoints_mean[c] = mean(cutpoints[,c]);

  // compute the mean for each cutpoint distance across subjects
  vector[N_resp_levels-2] jumps_mean; 

  for (j in 1:(N_resp_levels-2))
    jumps_mean[j] = mean(jumps[,j]);
}