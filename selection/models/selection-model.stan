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
  // the relationship between a {verb, frame} and a component
  matrix<lower=0,upper=1>[N_verb,N_component] verb_component;
  matrix<lower=0,upper=1>[N_frame,N_component] frame_component;

  // the subjects cutpoint center
  vector<lower=0>[N_subj] subj_center;

  // the alpha and beta parameter for the jump distribution                        
  vector<lower=0>[N_resp_levels-1] subj_alpha;                      
  vector<lower=0>[N_resp_levels-1] subj_beta;

  // cutpoint distances for each subject
  matrix<lower=0>[N_subj,N_resp_levels-2] jumps;
}

transformed parameters {
  // compute the cutpoints by taking a cumulative sum
  matrix[N_resp_levels-1,N_subj] cutpoints;

  for (s in 1:N_subj) {
    for (c in 1:(N_resp_levels-1)) {
      if (c == 1) {
        cutpoints[c,s] = 0.0 - subj_center[s];
      } else {
        cutpoints[c,s] = cutpoints[c-1,s] + jumps[s,c-1] - subj_center[s];
      }
    }
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
  // sample the centers
  subj_center ~ gamma(subj_alpha[1], subj_beta[1]);

  // sample the cutpoint distances
  for (j in 1:(N_resp_levels-2))
    jumps[,j] ~ gamma(subj_alpha[j+1], subj_beta[j+1]);

  // sample the responses
  for (n in 1:N_resp) {
    resp[n] ~ ordered_logistic(mu[n], cutpoints);
  }
}