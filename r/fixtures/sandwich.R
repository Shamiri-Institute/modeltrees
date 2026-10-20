# Cluster-robust leaf covariances: sandwich::vcovCL (sandwich 3.1-3) on the
# leaf fits of clustered trees, one fixture per tree in tests/fixtures/vcovcl/.
# Sourced by r/generate_all.R after trees.R and extensions.R. Decisions and the
# limits of what partykit and sandwich expose are in docs/sandwich-fixtures.md.

# ---------------------------------------------------------------------------
# Data and trees

# Grouped data in the style of sim_cluster: 36 schools of 8 to 40 pupils (rows
# shuffled), a school-level random intercept u (sd 0.5; for the multinomial
# one per non-reference class), school-level zs and zt without effect,
# observation-level z1 (planted split at 5), z2 (nominal) and z3 (numeric), and
# an exposure for the count models. One response per model:
# y_pois poisson with offset log(exposure); y_nb negative binomial (theta 2)
# with the same offset; y_mn three classes lo < mid < hi (lo the reference);
# y_sl binary without separation, for the safe logit.
build_sim_cluster_models <- function() {
  g <- 36L
  school <- rep(seq_len(g), sample(8:40, g, replace = TRUE))
  n <- length(school)
  zs <- runif(g, 0, 10)
  zt <- factor(sample(c("urban", "rural", "peri"), g, replace = TRUE))
  u <- rnorm(g, 0, 0.5)[school]
  um <- rnorm(g, 0, 0.5)[school]
  uh <- rnorm(g, 0, 0.5)[school]
  d <- data.frame(
    school = school,
    x = rnorm(n),
    zs = zs[school],
    zt = zt[school],
    z1 = runif(n, 0, 10),
    z2 = factor(sample(c("a", "b", "c"), n, replace = TRUE)),
    z3 = rnorm(n),
    exposure = runif(n, 0.5, 2)
  )
  left <- d$z1 <= 5
  eta <- u + ifelse(left, 0.2 + 0.7 * d$x, 0.9 - 0.4 * d$x)
  d$y_pois <- rpois(n, d$exposure * exp(eta))
  d$y_nb <- rnbinom(n, mu = d$exposure * exp(eta), size = 2)
  em <- um + ifelse(left, 0.3 + 0.8 * d$x, -0.5 - 0.5 * d$x)
  eh <- uh + ifelse(left, -0.2 - 0.6 * d$x, 0.6 + 1.0 * d$x)
  p <- cbind(1, exp(em), exp(eh))
  p <- p / rowSums(p)
  v <- runif(n)
  lev <- c("lo", "mid", "hi")
  d$y_mn <- factor(lev[1L + (v > p[, 1L]) + (v > p[, 1L] + p[, 2L])], levels = lev)
  d$y_sl <- as.integer(runif(n) < plogis(u + ifelse(left, -0.5 + 1.2 * d$x, 0.5 - 0.8 * d$x)))
  d[sample(n), ]
}

sandwich_datasets <- function() {
  list(
    make_dataset("sim_cluster_models",
      "Simulated grouped data for the cluster-robust leaf covariances: 36 schools of 8 to 40 pupils with school-level random intercepts, a planted split at z1 = 5, and responses for poisson (y_pois, offset log(exposure)), negative binomial (y_nb), multinomial (y_mn) and safe logit without separation (y_sl). The column school holds the cluster id.",
      "simulated", 20261041L, build_sim_cluster_models)
  )
}

# Trees grown with cluster = school on sim_cluster_models. They live in
# tests/fixtures/cluster_trees/ (same format as trees/ and extension_trees/)
# so that the Python suites that glob trees/ keep their fixture counts.
all_cluster_tree_specs <- function() {
  zc <- "zs + zt + z1 + z2 + z3"
  nb_args <- list(epsilon = 1e-8, maxit = 25L)
  sl_args <- list(epsilon = 1e-8, maxit = 25L, separation_tol = 1e-6)
  mn_args <- list(maxit = 1000L, reltol = 1e-16, abstol = 1e-4)
  list(
    tree_spec("sim_cluster_models_glmtree_poisson_cluster", "sim_cluster_models", "glmtree",
      paste("y_pois ~ x |", zc),
      "Poisson counts with an offset, a school-level random intercept and cluster = school.",
      "one", family = "poisson(link = \"log\")", offset = "log(exposure)", cluster = "school",
      control = list(minsize = 40L)),
    tree_spec("sim_cluster_models_negbin_cluster", "sim_cluster_models", "mob",
      paste("y_nb ~ x |", zc),
      "Negative binomial counts with an offset, a school-level random intercept and cluster = school.",
      "one", fit = "negbinfit", fit_args = nb_args, offset = "log(exposure)", cluster = "school",
      control = list(minsize = 40L)),
    tree_spec("sim_cluster_models_safelogit_cluster", "sim_cluster_models", "mob",
      paste("y_sl ~ x |", zc),
      "Safe logit without separation, a school-level random intercept and cluster = school.",
      "one", fit = "safelogitfit", fit_args = sl_args, cluster = "school",
      control = list(minsize = 40L)),
    tree_spec("sim_cluster_models_multinom_cluster", "sim_cluster_models", "mob",
      paste("y_mn ~ x |", zc),
      "Multinomial logit with school-level random intercepts per class and cluster = school.",
      "one", fit = "multinomfit", fit_args = mn_args, cluster = "school",
      control = list(minsize = 60L))
  )
}

# ---------------------------------------------------------------------------
# Leaf refits
#
# sandwich needs a fitted object of a class it has methods for. Every leaf is
# refitted on its rows with the standard R function, and the refit's
# coefficients are compared with the tree's (they must agree to 1e-10
# relative; the observed difference is recorded).

VCOVCL_MODELS <- list(
  sim_cluster_lmtree_cluster = list(model = "lm", tree_dir = "trees"),
  sim_cluster_glmtree_binomial_cluster = list(model = "glm_binomial", tree_dir = "trees"),
  sim_cluster_few_lmtree_cluster = list(model = "lm", tree_dir = "trees"),
  sim_cluster_models_glmtree_poisson_cluster = list(model = "glm_poisson", tree_dir = "cluster_trees"),
  sim_cluster_models_negbin_cluster = list(model = "negbin", tree_dir = "cluster_trees"),
  sim_cluster_models_safelogit_cluster = list(model = "safelogit", tree_dir = "cluster_trees"),
  sim_cluster_models_multinom_cluster = list(model = "multinom", tree_dir = "cluster_trees")
)

# The R expression used to refit a leaf. `leaf` is the data frame of the
# leaf's rows, in the fixture's `rows` order.
leaf_refit_text <- function(model, spec) {
  f <- spec$formula
  lhs_rhs <- sub("\\s*\\|.*$", "", f)
  switch(model,
    lm = sprintf("stats::lm(%s, data = leaf)", lhs_rhs),
    glm_binomial = sprintf("stats::glm(%s, family = binomial(link = \"logit\"), data = leaf, control = glm.control(epsilon = 1e-8, maxit = 25))", lhs_rhs),
    glm_poisson = sprintf("stats::glm(%s, family = poisson(link = \"log\"), offset = log(exposure), data = leaf, control = glm.control(epsilon = 1e-8, maxit = 25))", lhs_rhs),
    negbin = sprintf("MASS::glm.nb(%s + offset(log(exposure)), data = leaf, control = glm.control(epsilon = 1e-8, maxit = 25))", lhs_rhs),
    safelogit = sprintf("stats::glm(%s, family = binomial(link = \"logit\"), data = leaf, control = glm.control(epsilon = 1e-8, maxit = 25))", lhs_rhs),
    multinom = sprintf("nnet::multinom(%s, data = leaf, trace = FALSE, maxit = 1000, reltol = 1e-16, abstol = 1e-4)", lhs_rhs))
}

leaf_object_note <- function(model) {
  switch(model,
    lm = "The tree's leaf object is partykit's lm.fit result with a patched class; the leaf is refitted with stats::lm on the leaf's rows instead.",
    glm_binomial = "The tree's leaf object is partykit's glm.fit result with a patched class; the leaf is refitted with stats::glm on the leaf's rows instead.",
    glm_poisson = "The tree's leaf object is partykit's glm.fit result with a patched class; the leaf is refitted with stats::glm on the leaf's rows instead.",
    negbin = "negbinfit fits MASS::glm.nb(Y ~ 0 + X); the leaf is refitted with MASS::glm.nb on the leaf's rows with the formula interface, a class sandwich handles (estfun.glm, bread.glm with dispersion 1).",
    safelogit = "safelogitfit returns a list (its object is glm.fit output with a patched class). Without separation it equals glm.fit with binomial(), so the leaf is refitted with stats::glm(family = binomial()).",
    multinom = "multinomfit returns a list. The leaf is refitted with nnet::multinom (formula interface, same maxit, reltol and abstol), which sandwich handles through estfun.multinom and bread.default (nobs times vcov.multinom, the generalized inverse of nnet's analytic Hessian).")
}

# Model Hessian -d(sum estfun)/d theta written by hand (docs/api-design.md,
# 5.2), from the refit's working quantities, not from sandwich.
leaf_hessian <- function(model, obj, X) {
  switch(model,
    lm = crossprod(X),
    glm_binomial = , glm_poisson = , negbin = , safelogit =
      crossprod(X * sqrt(obj$weights)),
    multinom = {
      pr <- obj$fitted.values
      w <- as.numeric(obj$weights)
      K <- ncol(pr)
      p <- ncol(X)
      H <- matrix(0, (K - 1L) * p, (K - 1L) * p)
      for (j in 2L:K) for (l in 2L:K) {
        cw <- w * pr[, j] * ((j == l) - pr[, l])
        H[(j - 2L) * p + seq_len(p), (l - 2L) * p + seq_len(p)] <- crossprod(X, cw * X)
      }
      H
    })
}

# The sandwich of docs/api-design.md 5.1 with unit weights:
# V = H^-1 M H^-1, M = c sum_g s_g s_g', c = G/(G-1) [cadjust] (n-1)/(n-k) [HC1].
sandwich_by_hand <- function(H, ef, cluster, type, cadjust) {
  n <- nrow(ef)
  k <- ncol(ef)
  if (is.null(cluster)) cluster <- seq_len(n)
  G <- length(unique(cluster))
  S <- rowsum(ef, cluster, reorder = FALSE)
  adj <- (if (cadjust) G / (G - 1) else 1) * (if (type == "HC1") (n - 1) / (n - k) else 1)
  Hi <- solve(H)
  Hi %*% (adj * crossprod(S)) %*% Hi
}

rel_diff <- function(a, b) {
  a <- unname(a)
  b <- unname(b)
  if (!all(is.finite(a) == is.finite(b))) return(Inf)
  ok <- is.finite(b)
  if (!any(ok)) return(0)
  max(abs(a[ok] - b[ok])) / max(abs(b[ok]))
}

mat <- function(m) unname(m)

vcovcl_leaf <- function(nd, model, spec, data, cluster_col) {
  rows <- as.integer(nd$rows) + 1L
  leaf <- data[rows, , drop = FALSE]
  rownames(leaf) <- NULL
  refit_text <- leaf_refit_text(model, spec)
  obj <- eval(parse(text = refit_text)[[1L]], list(leaf = leaf), globalenv())

  tree_cf <- as.numeric(nd$coefficients)
  names(tree_cf) <- as.character(nd$coefficient_names)
  refit_cf <- if (model == "multinom") {
    b <- coef(obj)
    stats::setNames(as.vector(t(b)), as.vector(t(outer(rownames(b), colnames(b), paste, sep = ":"))))
  } else coef(obj)
  if (!identical(names(refit_cf), names(tree_cf))) stop(spec$name, ": refit coefficient names differ")
  cf_rel <- max(abs(refit_cf - tree_cf) / pmax(1, abs(tree_cf)))
  if (cf_rel > 1e-10) stop(spec$name, " node ", nd$id, ": refit coefficients differ from the tree's by ", cf_rel)

  # The tree's own scores (partykit's or the extension fit function's estfun).
  ef <- matrix(as.numeric(nd$estfun), nrow = NROW(nd$estfun))
  colnames(ef) <- as.character(nd$estfun_names)
  if (!identical(colnames(ef), names(tree_cf))) stop(spec$name, ": aliased columns not expected")
  ef_rel <- rel_diff(sandwich::estfun(obj), ef)

  X <- stats::model.matrix(obj)
  H <- leaf_hessian(model, obj, X)
  dimnames(H) <- list(names(tree_cf), names(tree_cf))
  n <- nrow(ef)
  k <- ncol(ef)
  cl <- leaf[[cluster_col]]
  G <- length(unique(cl))
  bread <- sandwich::bread(obj)
  bread_rel <- rel_diff(bread / n, solve(H))

  grid <- expand.grid(type = c("HC0", "HC1"), cadjust = c(TRUE, FALSE),
    clustered = c(TRUE, FALSE), stringsAsFactors = FALSE)
  checks <- numeric()
  vc <- lapply(seq_len(nrow(grid)), function(i) {
    g <- grid[i, ]
    clv <- if (g$clustered) cl else NULL
    V <- sandwich::vcovCL(obj, cluster = clv, type = g$type, cadjust = g$cadjust)
    Vh <- sandwich_by_hand(H, ef, clv, g$type, g$cadjust)
    d <- rel_diff(V, Vh)
    checks[i] <<- d
    # G <= k leaves: the meat has rank at most G - 1 (the summed scores are
    # zero at the fit), so V is singular; R returns it without warning.
    if (d > 1e-8) {
      stop(sprintf("%s node %d: vcovCL(type = %s, cadjust = %s, cluster = %s) differs from the sandwich by hand (%g)",
        spec$name, nd$id, g$type, g$cadjust, if (g$clustered) cluster_col else "NULL", d))
    }
    ev <- eigen(V, symmetric = TRUE, only.values = TRUE)$values
    list(
      cluster = if (g$clustered) cluster_col else NULL,
      type = g$type,
      cadjust = g$cadjust,
      n_clusters = if (g$clustered) G else n,
      r_call = sprintf("sandwich::vcovCL(fit, cluster = %s, type = \"%s\", cadjust = %s)",
        if (g$clustered) sprintf("leaf$%s", cluster_col) else "NULL", g$type, g$cadjust),
      matrix = mat(V),
      eigenvalues = arr(ev),
      vs_by_hand_max_rel = d
    )
  })

  list(
    id = nd$id,
    rows = arr(nd$rows),
    n = n,
    k = k,
    n_clusters = G,
    clusters_le_k = G <= k,
    coefficient_names = arr(names(tree_cf)),
    coefficients = arr(unname(tree_cf)),
    refit_call = refit_text,
    refit_coefficients = arr(unname(refit_cf)),
    refit_vs_tree_coefficients_max_rel = cf_rel,
    refit_estfun_vs_tree_estfun_max_rel = ef_rel,
    vcov_model = mat(stats::vcov(obj)),
    bread = mat(bread),
    hessian = mat(H),
    bread_over_n_vs_hessian_inverse_max_rel = bread_rel,
    vcovcl = vc
  )
}

# One leaf of a tree, every row put in a single cluster (G = 1), which is not a
# leaf of any clustered tree but documents what vcovCL returns there.
vcovcl_single_cluster <- function(nd, model, spec, data) {
  rows <- as.integer(nd$rows) + 1L
  leaf <- data[rows, , drop = FALSE]
  rownames(leaf) <- NULL
  obj <- eval(parse(text = leaf_refit_text(model, spec))[[1L]], list(leaf = leaf), globalenv())
  one <- rep(1L, nrow(leaf))
  # meatCL aggregates with apply(ef, 2, rowsum, cluster), which drops a single
  # cluster to a vector, so crossprod() then fails: R gives an error, not Inf.
  lapply(c(TRUE, FALSE), function(ca) {
    V <- tryCatch(sandwich::vcovCL(obj, cluster = one, type = "HC0", cadjust = ca),
      error = function(e) e)
    if (!inherits(V, "error")) stop(spec$name, ": vcovCL with one cluster no longer fails; update the fixture")
    list(node = nd$id, cluster = "all rows in one cluster", type = "HC0", cadjust = ca,
      r_call = sprintf("sandwich::vcovCL(fit, cluster = rep(1L, nrow(leaf)), type = \"HC0\", cadjust = %s)", ca),
      error = conditionMessage(V))
  })
}

vcovcl_fixture <- function(spec, tree_fx, ds) {
  cfg <- VCOVCL_MODELS[[spec$name]]
  leaves <- Filter(function(nd) nd$terminal, tree_fx$nodes)
  if (any(vapply(leaves, function(nd) is.null(nd$estfun), TRUE))) stop(spec$name, ": a leaf has no estfun")
  out_leaves <- lapply(leaves, vcovcl_leaf, model = cfg$model, spec = spec, data = ds$data,
    cluster_col = spec$cluster)
  if (cfg$model == "safelogit") {
    sep <- vapply(leaves, function(nd) nd$model_info$separation, "")
    if (any(sep != "none")) stop(spec$name, ": a safe logit leaf is separated")
  }
  fx <- list(
    fixture = paste0("vcovcl_", spec$name),
    description = sprintf("sandwich::vcovCL on every leaf of %s/%s.json.", cfg$tree_dir, spec$name),
    generator = "r/fixtures/sandwich.R",
    versions = c(fixture_versions(), list(MASS = utils::packageDescription("MASS")$Version,
      nnet = utils::packageDescription("nnet")$Version)),
    tree_fixture = file.path(cfg$tree_dir, paste0(spec$name, ".json")),
    dataset = spec$dataset,
    data_csv = file.path("data", paste0(spec$dataset, ".csv")),
    model = cfg$model,
    cluster = spec$cluster,
    object_construction = leaf_object_note(cfg$model),
    weights = "none: every tree here is grown without case weights, so the refits have unit weights and no zero-weight rows",
    estfun_source = "the tree fixture's node estfun (partykit's or the extension fit function's scores), used for the check by hand",
    hessian_definition = "-d(sum estfun)/d theta by hand: X'X (lm), X' diag(working weights of the last IRLS iteration) X (glm, negbin at the refit's theta, safe logit), the multinomial information sum_i w_i (diag(p_i) - p_i p_i') kron x_i x_i' over the non-reference classes, class-major",
    check = "every vcovCL matrix equals H^-1 (c sum_g s_g s_g') H^-1 with the tree's estfun and the Hessian by hand (docs/api-design.md 5.1) to 1e-8 relative to the largest entry; the generator stops otherwise",
    leaves = out_leaves
  )
  if (spec$name == "sim_cluster_glmtree_binomial_cluster") {
    fx$single_cluster <- vcovcl_single_cluster(leaves[[1L]], cfg$model, spec, ds$data)
  }
  fx
}
