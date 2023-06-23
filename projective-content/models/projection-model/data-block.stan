data {
  int<lower=0> N_resp;                                // number of responses
  int<lower=0> N_verb;                                // number of verbs
  int<lower=0> N_context;                             // number of contexts
  int<lower=0> N_subj;                                // number of subjects
  vector[N_verb] verb_mean;                           // the verb means inferred from a previous model fit
  vector[N_verb] verb_std;                            // the verb standard deviations inferred from a previous model fit
  vector[N_context] context_mean;                     // the context means inferred from the norming data
  vector[N_context] context_std;                      // the context standard deviations inferred from the norming data
  int<lower=1,upper=N_verb> verb[N_resp];             // verb corresponding to response n
  int<lower=1,upper=N_context> context[N_resp];       // context corresponding to response n
  int<lower=1,upper=N_subj> subj[N_resp];             // subject corresponding to response n
  vector<lower=0,upper=1>[N_resp] resp;               // bounded slider response   
}