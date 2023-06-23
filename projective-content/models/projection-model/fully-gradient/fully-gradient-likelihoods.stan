functions {
  real truncated_normal_lpdf(real x, real mu, real sigma, real a, real b) {
    return normal_lpdf(x | mu, sigma) - 
           log_diff_exp(normal_lcdf(b | mu, sigma), 
                        normal_lcdf(a | mu, sigma));
  }
  real log_lik_lpdf(real resp, real verb_prob, real context_prob, real sigma) {
    real prob_or = 1.0 - (1.0 - verb_prob) * (1.0 - context_prob);

    return truncated_normal_lpdf(resp | prob_or, sigma, 0, 1);
  }
}