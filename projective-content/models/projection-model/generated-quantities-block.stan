generated quantities {
  vector[N_verb] verb_prob = inv_logit(
    verb_intercept
  );

  vector[N_context] context_prob = inv_logit(
    context_intercept
  );
}