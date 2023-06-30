parameters {
  real<lower=0> verb_intercept_std;                   // the verb random intercept standard deviation
  vector[N_verb] verb_intercept_z;                    // the verb random intercepts z-score
  real<lower=0> context_intercept_std;                // the context random intercept standard deviation
  vector[N_context] context_intercept_z;              // the context random intercepts z-score
  real<lower=0> subj_intercept_verb_std;              // the subject random verb intercept standard deviation
  vector[N_subj] subj_intercept_verb_z;               // the subject random verb intercepts z-score
  real<lower=0> subj_intercept_context_std;           // the subject random context intercept standard deviation
  vector[N_subj] subj_intercept_context_z;            // the subject random context intercepts z-score
  real<lower=0,upper=1> sigma;                        // the standard deviation of the likelihood
}

transformed parameters {
  // verb parameters
  vector[N_verb] verb_intercept = verb_intercept_std * verb_intercept_z;

  // context parameters
  vector[N_context] context_intercept = context_intercept_std * context_intercept_z;

  // subject parameters
  vector[N_subj] subj_intercept_verb = subj_intercept_verb_std * subj_intercept_verb_z;
  vector[N_subj] subj_intercept_context = subj_intercept_context_std * subj_intercept_context_z;

  // log-likelihood
  vector[N_resp] log_lik;
  vector[N_resp] verb_prob_by_resp;
  vector[N_resp] context_prob_by_resp;

  for (n in 1:N_resp) {
    verb_prob_by_resp[n] = inv_logit(
      verb_intercept[verb[n]] + subj_intercept_verb[subj[n]]
    );
    context_prob_by_resp[n] = inv_logit(
      context_intercept[context[n]] + subj_intercept_context[subj[n]]
    );
    log_lik[n] = log_lik_lpdf(
      resp[n] | verb_prob_by_resp[n], context_prob_by_resp[n], sigma
    );
  }
}

model {
  // sample the verb intercepts
  verb_intercept_std ~ exponential(1);
  verb_intercept_z ~ std_normal();

  // sample the context intercepts
  context_intercept_std ~ exponential(1);
  context_intercept_z ~ std_normal();

  // sample the subject intercepts
  subj_intercept_verb_std ~ exponential(1);
  subj_intercept_verb_z ~ std_normal();

  subj_intercept_context_std ~ exponential(1);
  subj_intercept_context_z ~ std_normal();
  
  // sample the responses
  for (n in 1:N_resp)
    target += log_lik[n];
}