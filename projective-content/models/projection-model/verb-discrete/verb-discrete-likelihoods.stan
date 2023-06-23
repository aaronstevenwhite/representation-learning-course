functions {
  real truncated_normal_lpdf(real x, real mu, real sigma, real a, real b) {
    return normal_lpdf(x | mu, sigma) - 
           log_diff_exp(normal_lcdf(b | mu, sigma), 
                        normal_lcdf(a | mu, sigma));
  }
  real log_lik_lpdf(real resp, real verb_prob, real context_prob, real sigma) {
    return log_mix(
      verb_prob,
      truncated_normal_lpdf(resp | 1, sigma, 0, 1),
      truncated_normal_lpdf(resp | context_prob, sigma, 0, 1)
    );
  }
}