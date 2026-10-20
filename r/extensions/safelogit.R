# Safe binary logistic regression node model, as a partykit::mob() type 1 fit
# function. The thesis names safelogitModel, "binary logistic regression in
# presence of quasi-complete separation" (reference/trusch.md:938-939), but
# does not reproduce its code, and the mobtools package that held it is not
# available. The rules below are this project's specification, written to
# give the behaviour the thesis describes (trusch.md:2607, 2651): a node whose
# data are separated is split off and left terminal, because the ML estimator
# does not exist there. See docs/extensions-spec.md, section 3.
#
# The fitter is glm.fit with binomial(link = "logit"). On data without
# complete separation the result equals partykit's glmfit for the same family.
# "Safe" means:
#   1. glm.fit warnings (non-convergence, fitted probabilities 0 or 1) are
#      caught and returned in `warnings`, not raised.
#   2. A glm.fit error does not stop mob(): the fit returns NA coefficients,
#      objfun = Inf (so split search treats the candidate as inadmissible) and
#      no estfun (so a node fit becomes terminal).
#   3. Complete separation: if every positive-weight observation is fitted to
#      within separation_tol of its 0/1 response, the ML estimator does not
#      exist and the scores are numerically zero. estfun is not returned, so
#      mob() makes the node terminal without testing. coefficients and objfun
#      are glm.fit's values (objfun is close to its supremum, 0).
#   Quasi-complete separation (some but not all observations at the boundary)
#   is only reported in `separation`; the fit and its scores are kept, so the
#   instability tests can still split the separated observations off.

safelogitfit <- function(y, x = NULL, start = NULL, weights = NULL, offset = NULL,
                         cluster = NULL, ..., estfun = FALSE, object = FALSE,
                         epsilon = 1e-8, maxit = 25, separation_tol = 1e-6) {
  if (is.null(x)) {
    x <- matrix(1, nrow = NROW(y), ncol = 1L, dimnames = list(NULL, "(Intercept)"))
  }
  if (is.null(weights)) weights <- rep(1, NROW(y))
  # Same response coding as binomial()$initialize: a factor's first level is
  # failure, every other level success.
  y01 <- if (is.factor(y)) as.numeric(y != levels(y)[1L]) else as.numeric(y)
  if (!all(y01 %in% c(0, 1))) stop("safelogitfit needs a binary response")

  warns <- character()
  z <- tryCatch(
    withCallingHandlers(
      stats::glm.fit(x = x, y = y01, weights = weights, start = start, offset = offset,
        family = stats::binomial(link = "logit"),
        control = stats::glm.control(epsilon = epsilon, maxit = maxit)),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  if (inherits(z, "error")) {
    cf <- rep(NA_real_, ncol(x))
    names(cf) <- colnames(x)
    return(list(coefficients = cf, objfun = Inf, estfun = NULL, object = NULL,
      status = "error", error = conditionMessage(z), separation = NA_character_,
      n_boundary = NA_integer_, converged = FALSE, boundary = NA, warnings = unique(warns)))
  }

  cf <- z$coefficients
  pos <- weights > 0
  at_bound <- abs(y01 - z$fitted.values)[pos] < separation_tol
  separation <- if (all(at_bound)) "complete" else if (any(at_bound)) "quasi" else "none"

  rval <- list(
    coefficients = cf,
    # aic/2 - rank is the negative log-likelihood, as in partykit's glmfit.
    objfun = z$aic / 2 - z$rank,
    estfun = NULL,
    object = NULL,
    status = "ok",
    separation = separation,
    n_boundary = sum(at_bound),
    converged = z$converged,
    boundary = z$boundary,
    warnings = unique(warns)
  )

  # The fitted glm.fit result as a "glm" object, so sandwich and stats methods
  # (estfun, logLik, predict) work on it. model.matrix() reads z$x.
  class(z) <- c("glm", "lm")
  z$x <- x
  z$offset <- if (is.null(offset)) 0 else offset

  if (estfun && separation != "complete") {
    ef <- sandwich::estfun(z)
    colnames(ef) <- colnames(x)[!is.na(cf)]
    rownames(ef) <- NULL
    rval$estfun <- ef
  }
  if (object) rval$object <- z
  rval
}

# Per-observation weighted binomial log-likelihood for a 0/1 response.
safelogit_loglik_obs <- function(beta, y01, x, weights, offset = NULL) {
  eta <- drop(x %*% beta)
  if (!is.null(offset)) eta <- eta + offset
  weights * (y01 * stats::plogis(eta, log.p = TRUE) +
    (1 - y01) * stats::plogis(eta, lower.tail = FALSE, log.p = TRUE))
}
