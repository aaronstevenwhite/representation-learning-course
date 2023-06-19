data {
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_subj;                           // number of subjects
  int<lower=2> N_resp_levels;                    // number of possible likert scale acceptability judgment responses
  int<lower=1,upper=N_subj> subj[N_resp];        // subject who gave response n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // likert scale acceptability judgment responses 
}

parameters {
  real acc_mean;                                 // mean acceptability
  real<lower=0> subj_intercept_std;              // subject random intercept standard deviation
  vector[N_subj] subj_intercept;                 // subject random intercepts
  vector<lower=0>[N_resp_levels-2] jumps;        // the cutpoint distances
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
}

model {
  // sample the subject intercepts
  subj_intercept ~ normal(0, subj_intercept_std);

  // sample the cutpoints distances
  for (j in 1:(N_resp_levels-2))
    jumps[j] ~ gamma(2,1);

  // sample the responses
  for (n in 1:N_resp)
    resp[n] ~ ordered_logistic(
      acc_mean, cutpoints + subj_intercept[subj[n]]
    );
}

generated quantities {
  // compute the log-likelihood
  real log_lik[N_resp];
  
  for (n in 1:N_resp)
    log_lik[n] = ordered_logistic_lpmf(
      resp[n] | acc_mean, cutpoints + subj_intercept[subj[n]]
    );
}