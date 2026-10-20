# Negative binomial node model with unknown shape parameter theta, as a
# partykit::mob() type 1 fit function. This is the partykit counterpart of
# negbinModel in Rusch's thesis (reference/trusch.md:1101-1122), which wrapped
# an adapted copy of MASS::glm.nb. Here MASS::glm.nb is called directly.
#
# coefficients are the regression coefficients only. theta is estimated in
# every node but is not a coefficient and has no estfun column: the thesis
# takes the scores from sandwich's estfun.glm (trusch.md:1056-1066), which
# treats "Negative Binomial" as a known-dispersion family. See
# docs/extensions-spec.md, section 2.
#
# objfun is the negative log-likelihood at (coefficients, theta) by default.
# Split search in partykit sums objfun over the two children, each with its
# own theta. objective = "deviance" uses the residual deviance at the node's
# own theta instead, which is what party 1.0-1 compared (mob_control's
# objfun = deviance, reference/party/R/MOB.R:91-92) and which reproduces the
# thesis' quine tree (trusch.md:1356-1368).

negbinfit <- function(y, x = NULL, start = NULL, weights = NULL, offset = NULL,
                      cluster = NULL, ..., estfun = FALSE, object = FALSE,
                      epsilon = 1e-8, maxit = 25,
                      objective = c("loglik", "deviance")) {
  objective <- match.arg(objective)
  if (is.null(x)) {
    x <- matrix(1, nrow = NROW(y), ncol = 1L, dimnames = list(NULL, "(Intercept)"))
  }
  Y <- as.numeric(y)
  X <- x
  W <- if (is.null(weights)) rep(1, NROW(Y)) else weights
  O <- offset
  # glm.nb builds its model frame from these local variables. x = TRUE stores
  # the design matrix, which sandwich::estfun() then reads via model.matrix().
  z <- MASS::glm.nb(Y ~ 0 + X, weights = W, offset = O, start = start,
    control = stats::glm.control(epsilon = epsilon, maxit = maxit), x = TRUE)
  cf <- z$coefficients
  names(cf) <- colnames(x)

  rval <- list(
    coefficients = cf,
    objfun = if (objective == "loglik") -z$twologlik / 2 else z$deviance,
    estfun = NULL,
    object = NULL,
    # Per-node shape parameter and diagnostics. mob() keeps these in the node
    # info. theta is the value returned by glm.nb. theta_irls is the theta of
    # the last IRLS fit, which produced the coefficients and the working
    # weights: glm.nb updates theta once more after that fit, so the two
    # differ by less than the alternation tolerance.
    theta = z$theta,
    theta_se = z$SE.theta,
    theta_irls = negbin_family_theta(z$family),
    theta_warning = if (is.null(z$th.warn)) NULL else z$th.warn,
    converged = z$converged,
    iter = z$iter
  )

  if (estfun) {
    # sandwich's estfun.glm: working residuals times working weights times x,
    # dispersion 1 for "Negative Binomial". Aliased columns are dropped.
    ef <- sandwich::estfun(z)
    colnames(ef) <- colnames(x)[!is.na(cf)]
    rownames(ef) <- NULL
    rval$estfun <- ef
  }

  if (object) {
    z$coefficients <- cf
    rval$object <- z
  }
  rval
}

# theta of a MASS::negative.binomial family object.
negbin_family_theta <- function(family) {
  get(".Theta", envir = environment(family$variance))
}

# Per-observation weighted log-likelihood, as in glm.nb's internal loglik().
negbin_loglik_obs <- function(beta, theta, y, x, weights, offset = NULL) {
  eta <- drop(x %*% beta)
  if (!is.null(offset)) eta <- eta + offset
  mu <- exp(eta)
  weights * (lgamma(theta + y) - lgamma(theta) - lgamma(y + 1) + theta * log(theta) +
    y * log(mu + (y == 0)) - (theta + y) * log(theta + mu))
}

# Per-observation score for theta, as in theta.ml's internal score(). It is not
# part of estfun; fixtures record it so the decision to leave theta out of the
# instability tests can be revisited.
negbin_theta_score_obs <- function(beta, theta, y, x, weights, offset = NULL) {
  eta <- drop(x %*% beta)
  if (!is.null(offset)) eta <- eta + offset
  mu <- exp(eta)
  weights * (digamma(theta + y) - digamma(theta) + log(theta) + 1 -
    log(theta + mu) - (y + theta) / (mu + theta))
}
