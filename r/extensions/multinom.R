# Multinomial logit node model, as a partykit::mob() type 1 fit function,
# fitted by nnet::multinom. The thesis interfaced the mlogit package
# (mlogitModel, reference/trusch.md:939-941) and does not reproduce its code.
# See docs/extensions-spec.md, section 4.
#
# The response is a factor with levels l_1, ..., l_K. l_1 is the reference
# class. coefficients has (K - 1) * p entries named "<level>:<regressor>" in
# class-major order (all regressors of l_2, then l_3, ...), which is the order
# of vcov(multinom) and of sandwich's estfun.multinom. The full length is kept
# in every node: classes absent from a node get NA coefficients and no estfun
# columns, like aliased coefficients in glmfit.
#
# nnet's BFGS optimiser stops on a relative change in the objective. The
# default reltol = 1e-8 leaves gradients of order 1e-4, too loose for an
# oracle, so the defaults here are maxit = 1000 and reltol = 1e-16 (iterate
# until BFGS makes no further progress).

multinomfit <- function(y, x = NULL, start = NULL, weights = NULL, offset = NULL,
                        cluster = NULL, ..., estfun = FALSE, object = FALSE,
                        maxit = 1000, reltol = 1e-16, abstol = 1e-4) {
  if (!is.factor(y)) stop("multinomfit needs a factor response")
  if (!is.null(offset)) stop("multinomfit does not support an offset")
  if (is.null(x)) {
    x <- matrix(1, nrow = NROW(y), ncol = 1L, dimnames = list(NULL, "(Intercept)"))
  }
  if (is.null(weights)) weights <- rep(1, NROW(y))
  lev <- levels(y)
  p <- ncol(x)
  coef_names <- as.vector(t(outer(lev[-1L], colnames(x), paste, sep = ":")))
  cf <- rep(NA_real_, length(coef_names))
  names(cf) <- coef_names

  # Fit on the positive-weight rows. A class counts as present if it has
  # positive total weight in the node.
  pos <- weights > 0
  present <- lev[lev %in% as.character(y[pos])]
  status <- if (length(present) < 2L) {
    "single_class"
  } else if (present[1L] != lev[1L]) {
    "reference_absent"
  } else {
    "ok"
  }

  rval <- list(coefficients = cf, objfun = 0, estfun = NULL, object = NULL,
    status = status, classes_present = present, convergence = NA_integer_,
    value = NA_real_)

  if (status == "single_class") {
    # Every observation is in one class: the log-likelihood supremum is 0 and
    # no coefficient is identified.
    return(rval)
  }

  Yf <- factor(y[pos], levels = present)
  X <- x[pos, , drop = FALSE]
  W <- weights[pos]
  args <- list(Yf ~ 0 + X, weights = W, trace = FALSE, maxit = maxit,
    reltol = reltol, abstol = abstol, model = TRUE)
  if (!is.null(start) && status == "ok") args$Wts <- multinom_start_wts(start, lev, present, p)
  m <- do.call(nnet::multinom, args)
  rval$convergence <- m$convergence
  rval$value <- m$value
  rval$objfun <- m$value

  if (status == "reference_absent") {
    # The model is fitted with the first present class as reference, so its
    # coefficients cannot be expressed relative to l_1. The objective is the
    # supremum of the full model's log-likelihood; no coefficient or score is
    # returned.
    if (object) rval$object <- m
    return(rval)
  }

  # coef(multinom) is a vector for two classes and a class-by-regressor
  # matrix otherwise.
  b <- matrix(coef(m), nrow = length(present) - 1L, ncol = p)
  for (j in seq_along(present)[-1L]) {
    cf[paste(present[j], colnames(x), sep = ":")] <- b[j - 1L, ]
  }
  rval$coefficients <- cf

  if (estfun) {
    # sandwich's estfun.multinom on the positive-weight rows; zero-weight rows
    # get zero scores, as in glmfit.
    ef_pos <- sandwich::estfun(m)
    ef <- matrix(0, nrow = NROW(y), ncol = sum(!is.na(cf)),
      dimnames = list(NULL, names(cf)[!is.na(cf)]))
    ef[pos, ] <- ef_pos
    rval$estfun <- ef
  }
  if (object) rval$object <- m
  rval
}

# nnet weight vector for a multinom fit started at `start` (a full coefficient
# vector in multinomfit's order). Layout: one block of 1 + p weights per output
# unit (nnet bias, fixed at 0, then the regressors). With three or more present
# classes there is one unit per class and the reference unit is all zero; with
# two there is a single unit.
multinom_start_wts <- function(start, lev, present, p) {
  blocks <- lapply(present[-1L], function(l) {
    s <- start[(match(l, lev) - 2L) * p + seq_len(p)]
    s[is.na(s)] <- 0
    c(0, s)
  })
  if (length(present) == 2L) return(blocks[[1L]])
  c(rep(0, p + 1L), unlist(blocks))
}

# Per-observation weighted log-likelihood. beta is the full coefficient vector;
# NA entries (absent classes) are treated as -Inf intercepts, i.e. the class is
# left out of the softmax.
multinom_loglik_obs <- function(beta, y, x, weights) {
  lev <- levels(y)
  p <- ncol(x)
  B <- matrix(beta, nrow = length(lev) - 1L, ncol = p, byrow = TRUE)
  eta <- cbind(0, x %*% t(B))
  keep <- c(TRUE, !is.na(B[, 1L]))
  eta[, !keep] <- -Inf
  mx <- apply(eta, 1L, max)
  lse <- mx + log(rowSums(exp(eta - mx)))
  ll <- eta[cbind(seq_len(NROW(y)), as.integer(y))] - lse
  ifelse(weights == 0, 0, weights * ll)
}
