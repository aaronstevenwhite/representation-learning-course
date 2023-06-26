data {
  int<lower=0> N_verb;                           // number of verbs
  int<lower=0> N_frame;                          // number of frames
  int<lower=0> N_component;                      // number of components
  int<lower=0> N_subj;                           // number of subjects
  int<lower=0> N_resp;                           // number of responses
  int<lower=0> N_resp_levels;                    // number of ordinal response levels
  int<lower=1,upper=N_verb> verb[N_resp];        // the verb associated with respone n
  int<lower=1,upper=N_frame> frame[N_resp];      // the frame associated with respone n
  int<lower=1,upper=N_subj> subj[N_resp];        // the subject associated with respone n
  int<lower=1,upper=N_resp_levels> resp[N_resp]; // the response
}

parameters {
  // the auxiliary variables that determine the component means
  vector<lower=0,upper=1>[N_component] verb_component_prior_mean_aux;
  vector<lower=0,upper=1>[N_component] frame_component_prior_mean_aux;

  // the precision of the components
  vector<lower=0>[N_component] verb_component_prior_precision;
  vector<lower=0>[N_component] frame_component_prior_precision;

  // the relationship between a {verb, frame} and a component
  matrix<lower=0,upper=1>[N_verb,N_component] verb_component;
  matrix<lower=0,upper=1>[N_frame,N_component] frame_component;

  // the subject intercepts
  vector[N_subj] subject_intercept;

  // the cutpoints
  ordered[N_resp_levels-1] cutpoints;
}

transformed parameters {
  // the component means
  vector<lower=0,upper=1>[N_component] verb_component_prior_mean;
  vector<lower=0,upper=1>[N_component] frame_component_prior_mean;

  // declare the prior parameters alpha and beta for each verb component
  vector<lower=0>[N_component] verb_component_prior_alpha;
  vector<lower=0>[N_component] verb_component_prior_beta;

  // declare the prior parameters alpha and beta for each frame component
  vector<lower=0>[N_component] frame_component_prior_alpha;
  vector<lower=0>[N_component] frame_component_prior_beta;

  // set the first component mean to the first auxiliary variability
  // this is the first bit of stick broken from the interval
  verb_component_prior_mean[1] = verb_component_prior_mean_aux[1];
  frame_component_prior_mean[1] = frame_component_prior_mean_aux[1];

  // compute the first verb component alpha and beta
  verb_component_prior_alpha[1] = verb_component_prior_mean[1] *
                                  verb_component_prior_precision[1];
  verb_component_prior_beta[1] = (1.0 - verb_component_prior_mean[1]) *
                                  verb_component_prior_precision[1];

  // compute the first frame component alpha and beta
  frame_component_prior_alpha[1] = frame_component_prior_mean[1] *
                                    frame_component_prior_precision[1];
  frame_component_prior_beta[1] = (1.0 - frame_component_prior_mean[1]) *
                                  frame_component_prior_precision[1];


  // compute the remaining components
  for (c in 2:N_component) {
    verb_component_prior_mean[c] = verb_component_prior_mean_aux[c] * 
                                   verb_component_prior_mean[c-1];
    verb_component_prior_alpha[c] = verb_component_prior_mean[c] *
                                    verb_component_prior_precision[c];
    verb_component_prior_beta[c] = (1.0 - verb_component_prior_mean[c]) *
                                   verb_component_prior_precision[c];

    
    frame_component_prior_mean[c] = frame_component_prior_mean_aux[c] * 
                                    frame_component_prior_mean[c-1];
    frame_component_prior_alpha[c] = frame_component_prior_mean[c] *
                                     frame_component_prior_precision[c];
    frame_component_prior_beta[c] = (1.0 - frame_component_prior_mean[c]) *
                                    frame_component_prior_precision[c];
  }

  // component the verb-frame acceptability
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

  // compute the log-odds
  // used as a parameter of the ordered logistic
  vector[N_resp] mu;

  for (n in 1:N_resp) {
    mu[n] = logit(verb_frame[verb[n],frame[n]]);
  }
}

model {
  // sample the component mean auxiliary variable hyperpriors
  verb_component_prior_mean_aux ~ beta(alpha, 1);
  frame_component_prior_mean_aux ~ beta(alpha, 1);

  // sample the component precision hyperpriors
  verb_component_prior_precision ~ exponential(1);
  frame_component_prior_precision ~ exponential(1);

  // sample the component priors
  for (v in 1:N_verb)
    verb_component[v] ~ beta(verb_component_prior_alpha, verb_component_prior_beta);

  for (f in 1:N_frame)
    frame_component[f] ~ beta(frame_component_prior_alpha, frame_component_prior_beta);

  // sample the cutpoints
  cutpoints ~ normal(0, 1);

  // sample the responses
  for (n in 1:N_resp) {
    resp[n] ~ ordered_logistic(mu[n], cutpoints);
  }
}