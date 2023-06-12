data {
  int<lower=0> N_verb;
  int<lower=0> N_frame;
  int<lower=0> N_component;
  int<lower=0> N_subj;
  int<lower=0> N_resp;
  int<lower=0> N_resp_levels;
  int<lower=1,upper=N_verb> verb[N_resp];
  int<lower=1,upper=N_frame> frame[N_resp];
  int<lower=1,upper=N_subj> subj[N_resp];
  int<lower=1,upper=N_resp_levels> resp[N_resp];
}

parameters {
  vector<lower=0,upper=1>[N_component] verb_component_prior_aux;
  vector<lower=0,upper=1>[N_component] frame_component_prior_aux;

  vector<lower=0>[N_component] verb_component_prior_precision;
  vector<lower=0>[N_component] frame_component_prior_precision;

  matrix<lower=0,upper=1>[N_verb,N_component] verb_component;
  matrix<lower=0,upper=1>[N_frame,N_component] frame_component;
  real<lower=0> scale;
  vector[N_subj] subject_intercept;
  ordered[N_resp_levels-1] cutpoints;
}

transformed parameters {
  vector<lower=0,upper=1>[N_component] verb_component_prior_mean;
  vector<lower=0>[N_component] verb_component_prior_alpha;
  vector<lower=0>[N_component] verb_component_prior_beta;

  vector<lower=0,upper=1>[N_component] frame_component_prior_mean;
  vector<lower=0>[N_component] frame_component_prior_alpha;
  vector<lower=0>[N_component] frame_component_prior_beta;

  verb_component_prior_mean[1] = verb_component_prior_aux[1];
  verb_component_prior_alpha[1] = verb_component_prior_mean[1] *
                                  verb_component_prior_precision[1];
  verb_component_prior_beta[1] = (1.0 - verb_component_prior_mean[1]) *
                                  verb_component_prior_precision[1];

  frame_component_prior_mean[1] = frame_component_prior_aux[1];
  frame_component_prior_alpha[1] = frame_component_prior_mean[1] *
                                    frame_component_prior_precision[1];
  frame_component_prior_beta[1] = (1.0 - frame_component_prior_mean[1]) *
                                  frame_component_prior_precision[1];


  for (c in 2:N_component) {
    verb_component_prior_mean[c] = verb_component_prior_aux[c] * 
                                   verb_component_prior_mean[c-1];
    verb_component_prior_alpha[c] = verb_component_prior_mean[c] *
                                    verb_component_prior_precision[c];
    verb_component_prior_beta[c] = (1.0 - verb_component_prior_mean[c]) *
                                   verb_component_prior_precision[c];

    
    frame_component_prior_mean[c] = frame_component_prior_aux[c] * 
                                    frame_component_prior_mean[c-1];
    frame_component_prior_alpha[c] = frame_component_prior_mean[c] *
                                     frame_component_prior_precision[c];
    frame_component_prior_beta[c] = (1.0 - frame_component_prior_mean[c]) *
                                    frame_component_prior_precision[c];
  }

  matrix[N_verb,N_frame] verb_frame;

  for (v in 1:N_verb) {
    for (f in 1:N_frame) {
      verb_frame[v,f] = 1.0;
      for (c in 1:N_component) {
        verb_frame[v,f] *= 1.0 - verb_component[v,c] * frame_component[f,c];
      }
    }
  }
  verb_frame = 1.0 - verb_frame;

  vector[N_resp] mu;

  for (n in 1:N_resp) {
    mu[n] = scale * (verb_frame[verb[n],frame[n]] - 0.5) + subject_intercept[subj[n]];
  }
}

model {
  if (nonparametric_prior) {
    verb_component_prior_aux ~ beta(alpha, 1);
    frame_component_prior_aux ~ beta(alpha, 1);

    verb_component_prior_precision ~ exponential(1);
    frame_component_prior_precision ~ exponential(1);

    for (v in 1:N_verb)
      verb_component[v] ~ beta(verb_component_prior_alpha, verb_component_prior_beta);

    for (f in 1:N_frame)
      frame_component[f] ~ beta(frame_component_prior_alpha, frame_component_prior_beta);
  } else {
      for (v in 1:N_verb)
        verb_component[v] ~ beta(alpha, alpha);

      for (f in 1:N_frame)
        frame_component[f] ~ beta(alpha, alpha);
  }

  // sample the cutpoints
  cutpoints ~ normal(0, 1);

  // sample the responses
  for (n in 1:N_resp) {
    resp[n] ~ ordered_logistic(mu[n], cutpoints);
  }
}