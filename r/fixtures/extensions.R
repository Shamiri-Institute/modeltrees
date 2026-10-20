# Phase 4 fixtures: negative binomial, safe logit and multinomial node models
# (r/extensions/). Datasets, node-level fit fixtures with estfun checks, tree
# specs, and the replication of the thesis' quine tree. Sourced by
# r/generate_all.R after common.R, datasets.R and trees.R.

# ---------------------------------------------------------------------------
# Datasets

build_quine <- function() {
  env <- new.env()
  utils::data("quine", package = "MASS", envir = env)
  env$quine
}

# Overdispersed counts (negative binomial, theta = 2) with an exposure offset.
# z1 numeric (planted split at 0.5), z2 nominal noise, z3 numeric noise.
# y_null: no instability. y_one: intercept and slope of x change at z1 = 0.5.
# y_theta: the coefficients are constant but theta changes at z1 = 0.5 (0.7 on
# the left, 20 on the right), which the instability tests do not target.
# y_cluster: y_one plus a school-level random intercept (sd 0.5).
# w: positive integer case weights.
build_sim_negbin <- function() {
  n <- 600L
  g <- 30L
  d <- data.frame(
    school = sample(seq_len(g), n, replace = TRUE),
    x = rnorm(n),
    z1 = runif(n),
    z2 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    z3 = rnorm(n),
    exposure = runif(n, 0.5, 2)
  )
  u <- rnorm(g, 0, 0.5)[d$school]
  left <- d$z1 <= 0.5
  eta_null <- 0.5 + 0.5 * d$x
  eta_one <- ifelse(left, 0.2 + 0.8 * d$x, 1.0 - 0.3 * d$x)
  d$y_null <- rnbinom(n, mu = d$exposure * exp(eta_null), size = 2)
  d$y_one <- rnbinom(n, mu = d$exposure * exp(eta_one), size = 2)
  d$y_theta <- rnbinom(n, mu = d$exposure * exp(eta_null), size = ifelse(left, 0.7, 20))
  d$y_cluster <- rnbinom(n, mu = d$exposure * exp(u + eta_one), size = 2)
  d$w <- sample(1:3, n, replace = TRUE)
  d
}

# Binary responses with separated regions. z2 == "g4" has y = 0 throughout
# (a constant response, like the thesis' non-voters, trusch.md:2607). Where
# z1 > 0.65 (and z2 != "g4") y = 1(x > 0), completely separated by x.
# Elsewhere y is logistic in x. z3 is numeric noise.
# y_plain: logistic everywhere with a planted split at z1 = 0.5 and no
# separation, for comparing safelogitfit with glmtree.
build_sim_separation <- function() {
  n <- 500L
  d <- data.frame(
    x = rnorm(n),
    z1 = runif(n),
    z2 = factor(sample(c("g1", "g2", "g3", "g4"), n, replace = TRUE)),
    z3 = rnorm(n)
  )
  u <- runif(n)
  y <- as.integer(u < plogis(-0.3 + 0.8 * d$x))
  sep <- d$z1 > 0.65
  y[sep] <- as.integer(d$x[sep] > 0)
  y[d$z2 == "g4"] <- 0L
  d$y <- y
  d$y_plain <- as.integer(u < plogis(ifelse(d$z1 <= 0.5, -0.5 + 1.2 * d$x, 0.5 - 0.8 * d$x)))
  # Indicator of the constant-response group, used as a regressor in the
  # quasi-separation node fit.
  d$d4 <- as.numeric(d$z2 == "g4")
  d
}

# Three-class responses lo < mid < hi (an unordered factor; lo is the
# reference). z1 numeric (planted split at 0.5), z2 nominal noise, z3 numeric
# noise. y_null: no instability. y_one: all four coefficients change at
# z1 = 0.5. y_cluster: y_one's linear predictors plus school-level random
# intercepts (sd 0.5) for mid and hi. w0: integer weights 0 to 3 (about a
# quarter zero), used only in node-level fits.
build_sim_multinom <- function() {
  n <- 900L
  g <- 30L
  d <- data.frame(
    school = sample(seq_len(g), n, replace = TRUE),
    x = rnorm(n),
    z1 = runif(n),
    z2 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    z3 = rnorm(n)
  )
  lev <- c("lo", "mid", "hi")
  draw <- function(eta_mid, eta_hi) {
    p <- cbind(1, exp(eta_mid), exp(eta_hi))
    p <- p / rowSums(p)
    u <- runif(n)
    factor(lev[1L + (u > p[, 1L]) + (u > p[, 1L] + p[, 2L])], levels = lev)
  }
  left <- d$z1 <= 0.5
  um <- rnorm(g, 0, 0.5)[d$school]
  uh <- rnorm(g, 0, 0.5)[d$school]
  em <- ifelse(left, 0.3 + 0.8 * d$x, -0.5 - 0.5 * d$x)
  eh <- ifelse(left, -0.2 - 0.6 * d$x, 0.6 + 1.0 * d$x)
  d$y_null <- draw(0.3 + 0.8 * d$x, -0.2 - 0.6 * d$x)
  d$y_one <- draw(em, eh)
  d$y_cluster <- draw(em + um, eh + uh)
  d$w0 <- sample(0:3, n, replace = TRUE)
  d
}

# Three classes lo < mid < hi (lo the reference) where the reference class is
# absent beyond z1 = 0.7: there only mid and hi occur, with their own
# coefficients. Below 0.7 all three classes occur. z2 nominal and z3 numeric
# noise. A split on z1 at or above the last lo observation gives a right
# child without the reference class.
build_sim_multinom_noref <- function() {
  n <- 600L
  d <- data.frame(
    x = rnorm(n),
    z1 = runif(n),
    z2 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    z3 = rnorm(n)
  )
  lev <- c("lo", "mid", "hi")
  right <- d$z1 > 0.7
  em <- ifelse(right, 0.4 + 0.6 * d$x, 0.2 + 0.7 * d$x)
  eh <- ifelse(right, -0.3 - 0.8 * d$x, -0.1 - 0.5 * d$x)
  p <- cbind(ifelse(right, 0, 1), exp(em), exp(eh))
  p <- p / rowSums(p)
  u <- runif(n)
  d$y <- factor(lev[1L + (u > p[, 1L]) + (u > p[, 1L] + p[, 2L])], levels = lev)
  d
}

extension_datasets <- function() {
  list(
    make_dataset("quine",
      "MASS::quine, days absent from school (Venables and Ripley). The thesis' negative binomial example (trusch.md:1339-1357).",
      "MASS", NULL, build_quine),
    make_dataset("sim_negbin",
      "Simulated overdispersed counts (theta = 2) with an exposure offset: no instability, a planted split at z1 = 0.5, a theta-only change, and a school-clustered version. Integer case weights w.",
      "simulated", 20261031L, build_sim_negbin),
    make_dataset("sim_separation",
      "Simulated binary data with a constant-response group (z2 = g4) and a region completely separated by x (z1 > 0.65); y_plain has no separation.",
      "simulated", 20261032L, build_sim_separation),
    make_dataset("sim_multinom",
      "Simulated three-class responses (lo, mid, hi): no instability, a planted split at z1 = 0.5, and a school-clustered version. Weights w0 in 0 to 3.",
      "simulated", 20261033L, build_sim_multinom),
    make_dataset("sim_multinom_noref",
      "Simulated three-class responses (lo, mid, hi) where the reference class lo never occurs for z1 > 0.7.",
      "simulated", 20261034L, build_sim_multinom_noref)
  )
}

# ---------------------------------------------------------------------------
# Node-level fits

FIT_FUNCTIONS <- c(negbin = "negbinfit", safelogit = "safelogitfit", multinom = "multinomfit")

node_fit_case <- function(name, model, dataset, description, response, rhs,
                          rows = NULL, weights = NULL, offset = NULL,
                          start = NULL, fit_args = list()) {
  list(name = name, model = model, dataset = dataset, description = description,
    response = response, rhs = rhs, rows = rows, weights = weights,
    offset = offset, start = start, fit_args = fit_args)
}

all_node_fit_cases <- function() {
  nb_args <- list(epsilon = 1e-8, maxit = 25L)
  sl_args <- list(epsilon = 1e-8, maxit = 25L, separation_tol = 1e-6)
  mn_args <- list(maxit = 1000L, reltol = 1e-16, abstol = 1e-4)
  list(
    node_fit_case("negbin_quine_full", "negbin", "quine",
      "All of quine, Days on Eth + Sex + Age + Lrn (7 coefficients).",
      "Days", "~ Eth + Sex + Age + Lrn", fit_args = nb_args),
    node_fit_case("negbin_quine_eth_a_intercept", "negbin", "quine",
      "Intercept only on the Eth = A rows (the thesis' node 2).",
      "Days", "~ 1", rows = quote(Eth == "A"), fit_args = nb_args),
    node_fit_case("negbin_sim_offset_weights", "negbin", "sim_negbin",
      "y_one on x with offset log(exposure) and integer case weights w, rows with z1 <= 0.5.",
      "y_one", "~ x", rows = quote(z1 <= 0.5), weights = "w",
      offset = "log(exposure)", fit_args = nb_args),
    node_fit_case("negbin_sim_start", "negbin", "sim_negbin",
      "y_one on x with offset, starting values c(0, 0) as passed by split search with restart = FALSE.",
      "y_one", "~ x", offset = "log(exposure)", start = c(0, 0), fit_args = nb_args),
    node_fit_case("negbin_quine_deviance", "negbin", "quine",
      "As negbin_quine_eth_a_intercept with objective = \"deviance\" (party 1.0-1's split criterion).",
      "Days", "~ 1", rows = quote(Eth == "A"),
      fit_args = c(nb_args, list(objective = "deviance"))),
    node_fit_case("safelogit_none", "safelogit", "sim_separation",
      "No separation (z1 <= 0.65, z2 != g4). Must equal partykit's glmfit with binomial().",
      "y", "~ x", rows = quote(z1 <= 0.65 & z2 != "g4"), fit_args = sl_args),
    node_fit_case("safelogit_quasi", "safelogit", "sim_separation",
      "Quasi-complete separation: regressors 1, x, d4 on z1 <= 0.65; every d4 = 1 row has y = 0, so the d4 coefficient diverges. Scores are returned.",
      "y", "~ x + d4", rows = quote(z1 <= 0.65), fit_args = sl_args),
    node_fit_case("safelogit_complete", "safelogit", "sim_separation",
      "Complete separation by x (z1 > 0.65, z2 != g4). No estfun.",
      "y", "~ x", rows = quote(z1 > 0.65 & z2 != "g4"), fit_args = sl_args),
    node_fit_case("safelogit_constant", "safelogit", "sim_separation",
      "Constant response y = 0 (z2 = g4). No estfun.",
      "y", "~ x", rows = quote(z2 == "g4"), fit_args = sl_args),
    node_fit_case("multinom_full", "multinom", "sim_multinom",
      "All three classes, y_one on x, all rows.",
      "y_one", "~ x", fit_args = mn_args),
    node_fit_case("multinom_weights_zero", "multinom", "sim_multinom",
      "Weights w0 with zeros: zero-weight rows are left out of the fit and get zero scores.",
      "y_one", "~ x", weights = "w0", fit_args = mn_args),
    node_fit_case("multinom_start", "multinom", "sim_multinom",
      "Starting values rep(0.1, 4) as passed by split search with restart = FALSE.",
      "y_one", "~ x", start = rep(0.1, 4), fit_args = mn_args),
    node_fit_case("multinom_absent_class", "multinom", "sim_multinom",
      "Class hi absent (rows with y_one != hi and z1 <= 0.5): hi coefficients are NA and have no estfun columns.",
      "y_one", "~ x", rows = quote(y_one != "hi" & z1 <= 0.5), fit_args = mn_args),
    node_fit_case("multinom_reference_absent", "multinom", "sim_multinom",
      "Reference class lo absent: objective only, all coefficients NA, no estfun.",
      "y_one", "~ x", rows = quote(y_one != "lo"), fit_args = mn_args),
    node_fit_case("multinom_single_class", "multinom", "sim_multinom",
      "Only class mid present: objective 0, all coefficients NA, no estfun.",
      "y_one", "~ x", rows = quote(y_one == "mid"), fit_args = mn_args)
  )
}

# Per-observation weighted log-likelihood at the fit, for the finite-difference
# check. For negbin the scores belong to the last IRLS fit, so theta_irls is
# used (see negbin.R).
node_loglik_fun <- function(model, fit, y, x, w, off) {
  switch(model,
    negbin = function(b) negbin_loglik_obs(b, fit$theta_irls, y, x, w, off),
    safelogit = function(b) safelogit_loglik_obs(b, if (is.factor(y)) as.numeric(y != levels(y)[1L]) else y, x, w, off),
    multinom = function(b) {
      full <- fit$coefficients
      full[!is.na(full)] <- b
      multinom_loglik_obs(full, y, x, w)
    })
}

# Analytic scores at the returned coefficients, written by hand. For multinom
# they equal sandwich's estfun.multinom exactly. For negbin and safelogit,
# sandwich's estfun.glm (like partykit's glmfit) multiplies the final working
# residuals by glm.fit's working weights, which glm.fit computes from the
# linear predictor before its last update, so the two agree only to the IRLS
# convergence tolerance.
node_estfun_by_hand <- function(model, fit, y, x, w, off) {
  cf <- fit$coefficients
  b <- cf[!is.na(cf)]
  switch(model,
    negbin = {
      eta <- drop(x[, !is.na(cf), drop = FALSE] %*% b) + (if (is.null(off)) 0 else off)
      mu <- exp(eta)
      th <- fit$theta_irls
      w * (y - mu) / (1 + mu / th) * x[, !is.na(cf), drop = FALSE]
    },
    safelogit = {
      y01 <- if (is.factor(y)) as.numeric(y != levels(y)[1L]) else y
      eta <- drop(x[, !is.na(cf), drop = FALSE] %*% b) + (if (is.null(off)) 0 else off)
      w * (y01 - plogis(eta)) * x[, !is.na(cf), drop = FALSE]
    },
    multinom = {
      lev <- levels(y)
      p <- ncol(x)
      B <- matrix(cf, nrow = length(lev) - 1L, ncol = p, byrow = TRUE)
      keep <- !is.na(B[, 1L])
      eta <- cbind(0, x %*% t(B))
      eta[, c(FALSE, !keep)] <- -Inf
      pr <- exp(eta - apply(eta, 1L, max))
      pr <- pr / rowSums(pr)
      cols <- lapply(which(keep), function(j) {
        w * (as.numeric(as.integer(y) == j + 1L) - pr[, j + 1L]) * x
      })
      do.call(cbind, cols)
    })
}

# Central finite differences of the per-observation log-likelihood, one row
# per observation.
fd_scores <- function(llfun, b) {
  h <- 1e-5 * pmax(1, abs(b))
  vapply(seq_along(b), function(j) {
    bp <- b
    bm <- b
    bp[j] <- b[j] + h[j]
    bm[j] <- b[j] - h[j]
    (llfun(bp) - llfun(bm)) / (2 * h[j])
  }, numeric(length(llfun(b))))
}

rel_max <- function(a, b) max(abs(a - b)) / max(1, max(abs(b)))

node_fit_fixture <- function(case, ds) {
  d <- ds$data
  rows <- if (is.null(case$rows)) seq_len(nrow(d)) else which(eval(case$rows, d))
  dd <- d[rows, , drop = FALSE]
  x <- stats::model.matrix(stats::as.formula(case$rhs), dd)
  attr(x, "assign") <- NULL
  attr(x, "contrasts") <- NULL
  rownames(x) <- NULL
  y <- dd[[case$response]]
  w <- if (is.null(case$weights)) rep(1, nrow(dd)) else as.numeric(dd[[case$weights]])
  off <- if (is.null(case$offset)) NULL else eval(parse(text = case$offset)[[1L]], dd)
  fitfun <- get(FIT_FUNCTIONS[[case$model]])
  args <- c(list(y = y, x = x, start = case$start, weights = w, offset = off,
    estfun = TRUE, object = FALSE), case$fit_args)
  fit <- do.call(fitfun, args)
  cf <- fit$coefficients
  ef <- fit$estfun

  # Relative differences are max |a - b| / max(1, max |b|). colsums are
  # relative to colSums(abs(estfun)).
  checks <- list()
  if (!is.null(ef)) {
    hand <- node_estfun_by_hand(case$model, fit, y, x, w, off)
    llfun <- node_loglik_fun(case$model, fit, y, x, w, off)
    fd <- fd_scores(llfun, unname(cf[!is.na(cf)]))
    cs <- colSums(ef)
    checks$estfun_vs_analytic_max_rel <- rel_max(ef, hand)
    checks$fd_vs_analytic_max_rel <- rel_max(fd, hand)
    checks$fd_vs_estfun_max_rel <- rel_max(ef, fd)
    checks$fd_step <- "1e-5 * max(1, |b_j|), central differences, per observation"
    checks$estfun_colsums <- arr(unname(cs))
    checks$estfun_colsums_max_rel <- max(abs(cs) / pmax(1, colSums(abs(ef))))
    exact_tol <- if (case$model == "multinom") 1e-10 else 1e-6
    # Under quasi-complete separation the MLE does not exist: the score along
    # the diverging coefficient is minus the sum of the boundary fitted
    # probabilities (about 2e-6 when glm.fit stops), and reaches 0 only in the
    # limit.
    sum_tol <- if (identical(fit$separation, "quasi")) 1e-5 else 1e-6
    if (checks$estfun_vs_analytic_max_rel > exact_tol) stop(case$name, ": estfun differs from the analytic scores")
    if (checks$fd_vs_analytic_max_rel > 1e-7) stop(case$name, ": analytic scores differ from finite differences")
    if (checks$fd_vs_estfun_max_rel > 1e-6) stop(case$name, ": estfun differs from finite-difference scores")
    if (checks$estfun_colsums_max_rel > sum_tol) stop(case$name, ": estfun does not sum to zero")
  }
  # The objective must equal the summed log-likelihood (negated).
  if (case$model == "negbin" && !identical(case$fit_args$objective, "deviance")) {
    ll <- sum(negbin_loglik_obs(unname(cf), fit$theta, y, x, w, off))
    checks$objfun_vs_loglik_abs <- abs(fit$objfun + ll)
    ts <- negbin_theta_score_obs(unname(cf), fit$theta, y, x, w, off)
    checks$theta_score_sum <- sum(ts)
  }
  if (case$model == "safelogit" && fit$status == "ok") {
    y01 <- if (is.factor(y)) as.numeric(y != levels(y)[1L]) else y
    checks$objfun_vs_loglik_abs <- abs(fit$objfun + sum(safelogit_loglik_obs(
      ifelse(is.na(cf), 0, cf), y01, x, w, off)))
  }
  if (case$model == "multinom" && fit$status == "ok") {
    checks$objfun_vs_loglik_abs <- abs(fit$objfun + sum(multinom_loglik_obs(cf, y, x, w)))
  }
  if (!is.null(checks$objfun_vs_loglik_abs) && checks$objfun_vs_loglik_abs > 1e-8 * max(1, abs(fit$objfun))) {
    stop(case$name, ": objfun is not the negative log-likelihood")
  }
  # Without separation, safelogitfit must reproduce partykit's glmfit.
  if (case$model == "safelogit" && fit$separation == "none") {
    ref <- partykit:::glmfit(y = y, x = x, weights = w, offset = off,
      family = stats::binomial(), estfun = TRUE)
    if (!identical(unname(ref$coefficients), unname(cf)) || !identical(ref$objfun, fit$objfun) ||
        !identical(unname(ref$estfun), unname(ef))) {
      stop(case$name, ": safelogitfit differs from partykit's glmfit without separation")
    }
    checks$identical_to_partykit_glmfit <- TRUE
  }

  extras <- fit[setdiff(names(fit), c("coefficients", "objfun", "estfun", "object"))]
  list(
    fixture = case$name,
    description = case$description,
    generator = "r/fixtures/extensions.R",
    model = case$model,
    fit_function = FIT_FUNCTIONS[[case$model]],
    r_call = sprintf("%s(y = %s, x = model.matrix(%s, data), start = %s, weights = %s, offset = %s, %s, estfun = TRUE)",
      FIT_FUNCTIONS[[case$model]], case$response, case$rhs, deparse_value(case$start),
      if (is.null(case$weights)) "rep(1, n)" else case$weights,
      if (is.null(case$offset)) "NULL" else case$offset,
      paste(sprintf("%s = %s", names(case$fit_args), vapply(case$fit_args, deparse_value, "")), collapse = ", ")),
    versions = c(fixture_versions(), list(MASS = utils::packageDescription("MASS")$Version,
      nnet = utils::packageDescription("nnet")$Version)),
    dataset = ds$name,
    data_csv = file.path("data", paste0(ds$name, ".csv")),
    row_filter = if (is.null(case$rows)) NULL else deparse_value(case$rows),
    rows = arr(rows - 1L),
    n = length(rows),
    response = case$response,
    response_levels = if (is.factor(y)) arr(levels(y)) else NULL,
    y = if (is.factor(y)) arr(as.character(y)) else arr(as.numeric(y)),
    x_names = arr(colnames(x)),
    x = unname(x),
    weights = arr(w),
    offset = if (is.null(off)) NULL else arr(off),
    offset_expression = case$offset,
    start = if (is.null(case$start)) NULL else arr(case$start),
    fit_args = case$fit_args,
    result = list(
      coefficient_names = arr(names(cf)),
      coefficients = arr(finite_or_na(cf)),
      objfun = if (is.finite(fit$objfun)) fit$objfun else NULL,
      estfun_names = if (is.null(ef)) NULL else arr(colnames(ef)),
      estfun = if (is.null(ef)) NULL else unname(ef),
      model_info = extras
    ),
    checks = checks
  )
}

# ---------------------------------------------------------------------------
# Trees

all_extension_tree_specs <- function() {
  nb_args <- list(epsilon = 1e-8, maxit = 25L)
  sl_args <- list(epsilon = 1e-8, maxit = 25L, separation_tol = 1e-6)
  mn_args <- list(maxit = 1000L, reltol = 1e-16, abstol = 1e-4)
  zn <- "z1 + z2 + z3"
  zq <- "Eth + Sex + Age + Lrn"
  list(
    tree_spec("quine_negbin_intercept", "quine", "mob", paste("Days ~ 1 |", zq),
      "The thesis' quine example (trusch.md:1356-1357) with partykit's objective, the negative log-likelihood. minsize 20 is party's minsplit default.",
      "several", fit = "negbinfit", fit_args = nb_args, control = list(minsize = 20L)),
    tree_spec("quine_negbin_intercept_deviance", "quine", "mob", paste("Days ~ 1 |", zq),
      "The thesis' quine example with objective = \"deviance\", party 1.0-1's split criterion. Reproduces the tree, test statistics and p-values printed in the thesis (trusch.md:1359-1368); checked by the generator.",
      "several", fit = "negbinfit", fit_args = c(nb_args, list(objective = "deviance")),
      control = list(minsize = 20L)),
    tree_spec("sim_negbin_nosplit", "sim_negbin", "mob", paste("y_null ~ x |", zn),
      "Negative binomial, no planted instability, with offset.", "none",
      fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)",
      control = list(minsize = 40L)),
    tree_spec("sim_negbin_onesplit", "sim_negbin", "mob", paste("y_one ~ x |", zn),
      "Negative binomial with a planted split at z1 = 0.5, with offset.", "one",
      fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)",
      control = list(minsize = 40L)),
    tree_spec("sim_negbin_theta_only", "sim_negbin", "mob", paste("y_theta ~ x |", zn),
      "Only theta changes at z1 = 0.5. theta is not among the tested parameters, so the tree does not split.",
      "none", fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)",
      control = list(minsize = 40L)),
    tree_spec("sim_negbin_weights", "sim_negbin", "mob", paste("y_one ~ x |", zn),
      "Negative binomial with positive integer case weights and offset. As frequency weights they raise the effective sample to 1171, and the tree splits beyond the planted z1 = 0.5. Some split-search fits hit theta.ml's iteration limit (see warnings).", "several",
      fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)", weights = "w",
      control = list(minsize = 40L)),
    tree_spec("sim_negbin_cluster", "sim_negbin", "mob", paste("y_cluster ~ x |", zn),
      "Negative binomial with a school-level random intercept and cluster = school.", "one",
      fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)", cluster = "school",
      control = list(minsize = 40L)),
    tree_spec("sim_separation_safelogit", "sim_separation", "mob", paste("y ~ x |", zn),
      "Safe logit on data with a constant-response group and a completely separated region. The separated nodes get no estfun and stop with stop_reason no_estfun.",
      "several", fit = "safelogitfit", fit_args = sl_args, control = list(minsize = 30L)),
    tree_spec("sim_separation_safelogit_plain", "sim_separation", "mob", paste("y_plain ~ x |", zn),
      "Safe logit without separation (planted split at z1 = 0.5). The generator checks that glmtree with binomial() grows the identical tree.",
      "one", fit = "safelogitfit", fit_args = sl_args, control = list(minsize = 30L)),
    tree_spec("sim_multinom_nosplit", "sim_multinom", "mob", paste("y_null ~ x |", zn),
      "Multinomial logit, three classes, no planted instability.", "none",
      fit = "multinomfit", fit_args = mn_args, control = list(minsize = 60L)),
    tree_spec("sim_multinom_onesplit", "sim_multinom", "mob", paste("y_one ~ x |", zn),
      "Multinomial logit, three classes, planted split at z1 = 0.5.", "one",
      fit = "multinomfit", fit_args = mn_args, control = list(minsize = 60L)),
    tree_spec("sim_multinom_cluster", "sim_multinom", "mob", paste("y_cluster ~ x |", zn),
      "Multinomial logit with school-level random intercepts and cluster = school.", "one",
      fit = "multinomfit", fit_args = mn_args, cluster = "school",
      control = list(minsize = 60L))
  )
}

# What the thesis prints for the quine tree (trusch.md:1359-1368 and the
# summaries at 1404-1470): node sizes, intercepts to 2-3 decimals, theta,
# instability statistics and 1 - p ("criterion").
THESIS_QUINE <- list(
  root = list(variable = "Eth", statistic = 11.36, criterion = 0.997),
  node2 = list(variable = "Age", statistic = 10.796, criterion = 0.962),
  terminal = list(
    list(levels = "Eth = A, Age in F0, F1", n = 33, intercept = 2.74, theta = 1.873),
    list(levels = "Eth = A, Age in F2, F3", n = 36, intercept = 3.28, theta = 1.5),
    list(levels = "Eth = N", n = 77, intercept = 2.5, theta = 0.9186)
  )
)

# Compare the deviance-objective quine fixture with the thesis printout at the
# printed precision. Stops on disagreement; returns the comparison.
thesis_quine_crosscheck <- function(fx) {
  nodes <- fx$nodes
  by_id <- function(id) nodes[[which(vapply(nodes, `[[`, 0L, "id") == id)]]
  root <- by_id(1L)
  n2 <- by_id(2L)
  terms <- Filter(function(nd) nd$terminal, nodes)
  stat_of <- function(nd, v) nd$test$statistic[match(v, nd$test$variables)]
  got <- list(
    root = list(variable = root$split$variable, statistic = stat_of(root, "Eth"),
      criterion = 1 - root$p_value),
    node2 = list(variable = n2$split$variable, statistic = stat_of(n2, "Age"),
      criterion = 1 - n2$p_value),
    terminal = lapply(terms, function(nd) list(n = nd$nobs,
      intercept = nd$coefficients[[1L]], theta = nd$model_info$theta))
  )
  ok <- got$root$variable == "Eth" && got$node2$variable == "Age" &&
    abs(got$root$statistic - 11.36) < 0.005 && abs(got$root$criterion - 0.997) < 0.0005 &&
    abs(got$node2$statistic - 10.796) < 0.0005 && abs(got$node2$criterion - 0.962) < 0.0005 &&
    identical(vapply(got$terminal, function(t) t$n, 0), c(33, 36, 77)) &&
    all(abs(vapply(got$terminal, function(t) t$intercept, 0) - c(2.74, 3.28, 2.5)) < 0.005) &&
    all(abs(vapply(got$terminal, function(t) t$theta, 0) - c(1.873, 1.5, 0.9186)) < 0.0005)
  if (!ok) stop("The deviance-objective quine tree does not reproduce the thesis")
  list(thesis = THESIS_QUINE, partykit = got, agree = ok)
}

# Without separation, safelogitfit must grow exactly the tree that glmtree
# grows with binomial(): same nodes, splits, coefficients, objectives, tests
# and scores, to the last bit. Stops otherwise.
check_safelogit_matches_glmtree <- function(spec, ds, instrumented) {
  glm_spec <- spec
  glm_spec$fun <- "glmtree"
  glm_spec$family <- "binomial(link = \"logit\")"
  glm_spec$fit <- NULL
  glm_spec$fit_args <- list()
  safe <- tree_fixture(spec, ds, instrumented)
  ref <- tree_fixture(glm_spec, ds, instrumented)
  drop_info <- function(nodes) lapply(nodes, function(nd) {
    nd$model_info <- NULL
    nd
  })
  # Compare what the fixtures would contain. glmfit's scores also carry the
  # model-frame attributes of x, which are not serialized.
  as_json <- function(v) jsonlite::toJSON(v, digits = I(17), na = "null", null = "null", auto_unbox = TRUE)
  if (!identical(as_json(drop_info(safe$nodes)), as_json(ref$nodes)) ||
      !identical(safe$fitted_node, ref$fitted_node)) {
    stop(spec$name, ": safelogitfit and glmtree grow different trees")
  }
  invisible(TRUE)
}

# The multinomial tree with a leaf without the reference class. It is written
# to tests/fixtures/reference_absent_trees/ in the extension tree format, not
# to extension_trees/: some Python tests assume no such leaf (see
# docs/sandwich-fixtures.md, section 3).
reference_absent_tree_specs <- function() {
  mn_args <- list(maxit = 1000L, reltol = 1e-16, abstol = 1e-4)
  zn <- "z1 + z2 + z3"
  list(
    tree_spec("sim_multinom_noref_onesplit", "sim_multinom_noref", "mob", paste("y ~ x |", zn),
      "Multinomial logit where the reference class lo is absent for z1 > 0.7. Split-search candidates at or above the last lo observation have a right child without lo, fitted with mid as reference (objective only, no coefficients or scores). The tree splits there, and that leaf stops with stop_reason no_estfun (model_info status reference_absent). See reference_absent.",
      "one", fit = "multinomfit", fit_args = mn_args, control = list(minsize = 60L))
  )
}

# For sim_multinom_noref_onesplit: every root split-search candidate refitted
# on its two sides with multinomfit, recording the classes present, which side
# lacks the reference class and each side's objective. The sum must equal the
# instrumented split-search objective (partykit fits the subsets the same
# way); the generator stops otherwise. Also checks that the tree has a leaf
# without the reference class.
reference_absent_check <- function(spec, fx, ds) {
  d <- ds$data
  root <- fx$nodes[[1L]]
  ss <- root$split_search
  z <- d[[ss$variable]][root$rows + 1L]
  y <- d[[fx$response]][root$rows + 1L]
  x <- stats::model.matrix(~ x, d[root$rows + 1L, , drop = FALSE])
  ref <- levels(y)[1L]
  fit_side <- function(rows) {
    do.call(multinomfit, c(list(y = y[rows], x = x[rows, , drop = FALSE]), spec$fit_args))
  }
  cands <- as.numeric(ss$candidates)
  obj <- as.numeric(ss$objective)
  out <- lapply(seq_along(cands), function(i) {
    left <- z <= cands[i]
    present <- function(rows) as.character(levels(y)[levels(y) %in% as.character(y[rows])])
    base <- list(candidate = cands[i], objective = if (is.na(obj[i])) NULL else obj[i],
      n_left = sum(left), n_right = sum(!left),
      classes_left = arr(present(left)), classes_right = arr(present(!left)))
    if (is.na(obj[i])) return(base)
    fl <- fit_side(left)
    fr <- fit_side(!left)
    c(base, list(
      status_left = fl$status, status_right = fr$status,
      objective_left = fl$objfun, objective_right = fr$objfun))
  })
  adm <- Filter(function(o) !is.null(o$objective), out)
  diffs <- vapply(adm, function(o) abs(o$objective_left + o$objective_right - o$objective) /
    max(1, abs(o$objective)), 0)
  if (max(diffs) > 1e-10) stop(spec$name, ": candidate side objectives do not add up to the split-search objective")
  noref <- Filter(function(o) identical(o$status_right, "reference_absent") ||
    identical(o$status_left, "reference_absent"), adm)
  leaves <- Filter(function(nd) nd$terminal && identical(nd$model_info$status, "reference_absent"), fx$nodes)
  if (!length(noref) || !length(leaves)) stop(spec$name, ": no reference-absent candidate or leaf")
  chosen <- root$split$breaks
  list(
    description = paste("Root split search on", ss$variable, "refitted side by side with multinomfit.",
      "candidates lists every candidate in split_search order; admissible ones (objective not null)",
      "have each side's status and objective, whose sum equals the split-search objective",
      "(max relative difference in sides_vs_split_search_max_rel). status reference_absent",
      "means the side has no", ref, "observation: its objective is that of multinom fitted with",
      "the first present class as reference, and it has no coefficients or scores."),
    reference_class = ref,
    variable = ss$variable,
    n_admissible = length(adm),
    n_reference_absent_candidates = length(noref),
    first_reference_absent_candidate = noref[[1L]]$candidate,
    chosen_break = chosen,
    chosen_is_reference_absent = any(vapply(noref, function(o) o$candidate == chosen, TRUE)),
    sides_vs_split_search_max_rel = max(diffs),
    leaves_without_reference = lapply(leaves, function(nd) list(id = nd$id, n = nd$nobs,
      classes_present = arr(as.character(nd$model_info$classes_present)),
      objfun = nd$objfun, stop_reason = nd$stop_reason)),
    candidates = out
  )
}
