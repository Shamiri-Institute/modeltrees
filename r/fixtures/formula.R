# Formula-feature fixtures: trees grown through lmtree() and glmtree() on data
# with factor and logical regressors, a model without an intercept, logical,
# ordered and five-level nominal partitioning variables, missing values and
# one-part formulas, each with predict() on new rows. Sourced by
# r/generate_all.R after trees.R and predict.R; written to
# tests/fixtures/formula_trees/. docs/formula-fixtures.md records the
# decisions.

# ---------------------------------------------------------------------------
# Datasets

# A factor regressor f whose declared level order (d, b, a, c) is not
# alphabetical. Every response changes at z1 = 5; z2 is noise. y_gauss and
# y_bin depend on x and f, y_cell on f alone (cell means, for y ~ 0 + f),
# y_mean and y_count on nothing but the side of the split (one-part formulas).
# y_bin is a factor whose first level, stay, is the failure.
build_sim_formula_factor <- function() {
  n <- 600L
  lev <- c("d", "b", "a", "c")
  d <- data.frame(
    x = rnorm(n),
    f = factor(sample(lev, n, replace = TRUE), levels = lev),
    z1 = runif(n, 0, 10),
    z2 = rnorm(n)
  )
  left <- d$z1 <= 5
  fl <- as.character(d$f)
  eff_left <- c(d = 0, b = 1, a = -1, c = 0.5)[fl]
  eff_right <- c(d = 0, b = -1, a = 1.5, c = -0.5)[fl]
  e <- rnorm(n)
  d$y_gauss <- unname(ifelse(left, 1 + 1.5 * d$x + eff_left, -0.5 + 0.5 * d$x + eff_right)) + e
  eta <- unname(ifelse(left, -0.5 + d$x + eff_left, 0.5 - d$x + eff_right))
  d$y_bin <- factor(ifelse(runif(n) < plogis(eta), "leave", "stay"), levels = c("stay", "leave"))
  d$y_cell <- unname(ifelse(left, c(d = 1, b = 2, a = 0, c = 3)[fl], c(d = -1, b = 0, a = 2, c = 1)[fl])) +
    rnorm(n)
  d$y_mean <- ifelse(left, 1, -1) + rnorm(n)
  d$y_count <- rpois(n, exp(ifelse(left, 0.3, 1.2)))
  d
}

# A logical regressor b and a logical partitioning variable zb. y_lreg changes
# with z1 (zb is noise for it); y_lsplit changes with zb.
build_sim_formula_logical <- function() {
  n <- 500L
  d <- data.frame(
    x = rnorm(n),
    b = runif(n) < 0.4,
    z1 = runif(n, 0, 10),
    zb = runif(n) < 0.5
  )
  e <- rnorm(n)
  d$y_lreg <- ifelse(d$z1 <= 5, 1 + 2 * d$x + 1 * d$b, -1 + 0.5 * d$x - 1 * d$b) + e
  d$y_lsplit <- ifelse(d$zb, 1 + 2 * d$x, -1 - d$x) + e
  d
}

# An ordered partitioning variable zo whose declared order (low, mid, high,
# top) is not alphabetical, and a nominal one zn with five levels declared
# v, r, t, p, s. Planted: zo <= mid at the root, then zn in {r, p} for
# zo > mid. z1 is noise.
build_sim_formula_ordered <- function() {
  n <- 700L
  lo <- c("low", "mid", "high", "top")
  ln <- c("v", "r", "t", "p", "s")
  d <- data.frame(
    x = rnorm(n),
    zo = factor(sample(lo, n, replace = TRUE), levels = lo, ordered = TRUE),
    zn = factor(sample(ln, n, replace = TRUE), levels = ln),
    z1 = runif(n)
  )
  low <- as.integer(d$zo) <= 2L
  rp <- d$zn %in% c("r", "p")
  b0 <- ifelse(low, 1, ifelse(rp, -1, 0.5))
  b1 <- ifelse(low, 1.5, ifelse(rp, -1, 0.5))
  d$y <- b0 + b1 * d$x + rnorm(n)
  d
}

# Missing values in the response y, the regressor x, the numeric partitioning
# variable z1, the factor partitioning variable zf and the weights w (8 rows
# each, two rows with two missing values). The level e of the regressor f
# and the level q of zf occur only in rows that have a missing value, so the
# model frame drops them. Planted: z1 <= 5, then zf in one child.
build_sim_formula_na <- function() {
  n <- 500L
  d <- data.frame(
    x = rnorm(n),
    f = factor(sample(c("d", "b", "a", "c"), n, replace = TRUE), levels = c("d", "b", "a", "c", "e")),
    z1 = runif(n, 0, 10),
    zf = factor(sample(c("k", "j", "m"), n, replace = TRUE), levels = c("k", "j", "m", "q")),
    w = sample(1:3, n, replace = TRUE)
  )
  left <- d$z1 <= 5
  km <- d$zf %in% c("k", "m")
  eff <- c(d = 0, b = 0.5, a = -0.5, c = 1)[as.character(d$f)]
  d$y <- unname(ifelse(left, 1 + 2 * d$x, ifelse(km, -1 + 0.5 * d$x, 0.5 - d$x)) + eff) + rnorm(n)
  holes <- matrix(sample(n, 40L), nrow = 8L)
  d$y[holes[, 1L]] <- NA
  d$x[holes[, 2L]] <- NA
  d$z1[holes[, 3L]] <- NA
  d$zf[holes[, 4L]] <- NA
  d$w[holes[, 5L]] <- NA
  # Two rows with two missing values, and levels only in dropped rows.
  d$z1[holes[1L, 1L]] <- NA
  d$w[holes[1L, 2L]] <- NA
  d$f[holes[1:3, 1L]] <- "e"
  d$zf[holes[1:3, 2L]] <- "q"
  d
}

# sim_formula_logical with the logical zb stored as the double 0/1: the tree
# R grows when the logical partitioning variable is numeric.
logical_as_numeric <- function(src, name, description, column) {
  data <- src$data
  data[[column]] <- as.numeric(data[[column]])
  list(
    name = name,
    description = description,
    source = "simulated",
    seed = src$seed,
    code = sprintf("%s with the logical column %s replaced by as.numeric(%s)", src$name, column, column),
    data = data
  )
}

formula_datasets <- function() {
  logical <- make_dataset("sim_formula_logical",
    "Simulated gaussian data with a logical regressor b and a logical partitioning variable zb: y_lreg has a planted split at z1 = 5 (zb is noise), y_lsplit a planted split on zb.",
    "simulated", 20261062L, build_sim_formula_logical)
  na <- make_dataset("sim_formula_na",
    "Simulated gaussian data with missing values (NA) in y, x, z1, zf and the weights w; the levels f = e and zf = q occur only in rows with a missing value. Planted split at z1 = 5, then on zf.",
    "simulated", 20261064L, build_sim_formula_na)
  na$allow_na <- TRUE
  list(
    make_dataset("sim_formula_factor",
      "Simulated data with a factor regressor f declared in the order d, b, a, c; every response has a planted split at z1 = 5 (z2 is noise): gaussian y_gauss and binary y_bin (factor stay, leave) on x and f, cell means y_cell on f, and y_mean (gaussian) and y_count (poisson) on the split alone.",
      "simulated", 20261061L, build_sim_formula_factor),
    logical,
    logical_as_numeric(logical, "sim_formula_logical_numeric",
      "sim_formula_logical with the logical zb stored as the double 0/1. Same rows and other columns.",
      "zb"),
    make_dataset("sim_formula_ordered",
      "Simulated gaussian data with an ordered partitioning variable zo declared low, mid, high, top and a nominal one zn with five levels declared v, r, t, p, s. Planted split zo <= mid, then zn in {r, p} for zo > mid. z1 is noise.",
      "simulated", 20261063L, build_sim_formula_ordered),
    na
  )
}

# ---------------------------------------------------------------------------
# Specs

# case is the item of the task (docs/formula-fixtures.md, section 1).
formula_spec <- function(case, ..., outcome = "ok", oracle = NULL, predict_seed = NULL) {
  spec <- tree_spec(...)
  spec$case <- case
  spec$outcome <- outcome
  spec$oracle <- oracle
  spec$predict_seed <- predict_seed
  spec
}

all_formula_tree_specs <- function() {
  zf <- "z1 + z2"
  list(
    formula_spec(1L, "sim_formula_factor_lmtree_factor", "sim_formula_factor", "lmtree",
      paste("y_gauss ~ x + f |", zf),
      "A factor regressor f declared d, b, a, c (not alphabetical): treatment coding against d, coefficients (Intercept), x, fb, fa, fc.",
      "one", predict_seed = 20261161L),
    formula_spec(2L, "sim_formula_factor_glmtree_binomial_factor", "sim_formula_factor", "glmtree",
      paste("y_bin ~ x + f |", zf),
      "glmtree binomial with the factor regressor f and a factor response declared stay, leave (stay is the failure).",
      "one", family = "binomial(link = \"logit\")", predict_seed = 20261162L),
    formula_spec(3L, "sim_formula_factor_lmtree_noint", "sim_formula_factor", "lmtree",
      paste("y_cell ~ 0 + f |", zf),
      "No intercept: the factor regressor gets an indicator per level, fd, fb, fa, fc.",
      "one", predict_seed = 20261163L),
    formula_spec(4L, "sim_formula_logical_lmtree_logical", "sim_formula_logical", "lmtree",
      "y_lreg ~ x + b | z1 + zb",
      "A logical regressor b with an intercept (one column bTRUE) and a logical partitioning variable zb, which the instability tests treat as numeric and no split selects.",
      "one", predict_seed = 20261164L),
    formula_spec(4L, "sim_formula_logical_lmtree_logical_noint", "sim_formula_logical", "lmtree",
      "y_lreg ~ 0 + b + x | z1 + zb",
      "The logical regressor b without an intercept: columns bFALSE, bTRUE and x. The same fits as sim_formula_logical_lmtree_logical, reparametrized.",
      "one", predict_seed = 20261165L),
    formula_spec(4L, "sim_formula_logical_lmtree_logical_split", "sim_formula_logical", "lmtree",
      "y_lsplit ~ x | z1 + zb",
      "A planted split on the logical partitioning variable zb. partykit 1.3-0 stops: its split search treats a logical as categorical and kidids_split then requires an integer variable. No tree; the oracle for Python is sim_formula_logical_numeric_lmtree_split.",
      "none", outcome = "error", oracle = "sim_formula_logical_numeric_lmtree_split"),
    formula_spec(4L, "sim_formula_logical_numeric_lmtree_split", "sim_formula_logical_numeric", "lmtree",
      "y_lsplit ~ x | z1 + zb",
      "sim_formula_logical_lmtree_logical_split with zb stored as the double 0/1: the split zb <= 0 that R grows for the numeric variable.",
      "one", predict_seed = 20261166L),
    formula_spec(5L, "sim_formula_ordered_lmtree", "sim_formula_ordered", "lmtree",
      "y ~ x | zo + zn + z1",
      "An ordered partitioning variable zo declared low, mid, high, top and a nominal zn with five levels declared v, r, t, p, s: splits zo <= mid and zn in {r, p}.",
      "several", predict_seed = 20261167L),
    formula_spec(6L, "sim_formula_na_lmtree_weights", "sim_formula_na", "lmtree",
      "y ~ x + f | z1 + zf",
      "Missing values in y, x, z1, zf and the weights: model.frame's na.omit drops the rows (na_action), and the levels f = e and zf = q, present only in dropped rows, are dropped from the model frame.",
      "several", weights = "w", predict_seed = 20261168L),
    formula_spec(7L, "sim_formula_factor_lmtree_onepart", "sim_formula_factor", "lmtree",
      paste("y_mean ~", zf),
      "A one-part formula: lmtree expands y_mean ~ z1 + z2 to y_mean ~ 1 | z1 + z2, an intercept-only model.",
      "one", predict_seed = 20261169L),
    formula_spec(7L, "sim_formula_factor_glmtree_poisson_onepart", "sim_formula_factor", "glmtree",
      paste("y_count ~", zf),
      "A one-part formula for glmtree poisson: y_count ~ 1 | z1 + z2.",
      "one", family = "poisson(link = \"log\")", predict_seed = 20261170L)
  )
}

# ---------------------------------------------------------------------------
# Fixtures

# A named list that serializes as a JSON object also when empty.
named_list <- function(x) if (length(x)) x else stats::setNames(list(), character())

# Regressor and partitioning variable names of a grown tree, from its
# expanded formula.
tree_parts <- function(tree) {
  f <- Formula::Formula(tree$info$formula)
  list(
    regressors = all.vars(formula(f, lhs = 0L, rhs = 1L)),
    partitioning = all.vars(formula(f, lhs = 0L, rhs = 2L))
  )
}

model_frame_info <- function(tree, ds) {
  mf <- tree$data
  removed <- attr(mf, "na.action")
  removed <- if (is.null(removed)) integer() else as.integer(removed)
  input_rows <- setdiff(seq_len(nrow(ds$data)), removed)
  if (!identical(as.integer(rownames(mf)), input_rows)) stop("model frame rows are not the input rows without na.action")
  vars <- setdiff(names(mf), c("(weights)", "(offset)", "(cluster)"))
  classes <- vapply(vars, function(v) {
    x <- mf[[v]]
    if (is.ordered(x)) "ordered" else if (is.factor(x)) "factor" else class(x)[1L]
  }, "")
  facs <- vars[vapply(vars, function(v) is.factor(mf[[v]]), TRUE)]
  list(
    na_action = list(
      class = if (length(removed)) class(attr(mf, "na.action")) else NULL,
      n_input = nrow(ds$data),
      n_used = nrow(mf),
      removed_rows = arr(removed - 1L),
      note = "removed_rows are 0-based indices into the data CSV, from the na.action attribute of the model frame; every other row index in this file refers to the remaining rows"
    ),
    model_frame = list(
      columns = arr(names(mf)),
      classes = as.list(classes),
      levels = named_list(stats::setNames(lapply(facs, function(v) arr(levels(mf[[v]]))), facs)),
      dropped_levels = named_list(stats::setNames(lapply(facs, function(v) {
        arr(setdiff(levels(ds$data[[v]]), levels(mf[[v]])))
      }), facs))
    )
  )
}

# Columns of newdata for JSON: factor labels, logicals as true and false,
# numbers. The new rows have no missing values.
formula_json_column <- function(v) {
  stopifnot(!anyNA(v))
  if (is.factor(v)) return(arr(as.character(v)))
  arr(v)
}

# New rows for predict(): regressors, response and weights from one training
# row, partitioning variables from another, both drawn from the rows the tree
# used that avoid one level of each factor (the first level of a factor
# regressor, the last of a factor partitioning variable) and have a logical
# regressor TRUE. The new rows keep the data's declared levels.
formula_newdata <- function(spec, tree, ds, n_new = 12L) {
  parts <- tree_parts(tree)
  mf <- tree$data
  used <- as.integer(rownames(mf))
  data <- ds$data
  ok <- rep(TRUE, length(used))
  excluded <- list()
  for (v in parts$regressors) {
    x <- mf[[v]]
    if (is.factor(x)) {
      excluded[[v]] <- levels(x)[1L]
      ok <- ok & x != levels(x)[1L]
    } else if (is.logical(x)) {
      excluded[[v]] <- "FALSE"
      ok <- ok & x
    }
  }
  for (v in parts$partitioning) {
    x <- mf[[v]]
    if (is.factor(x)) {
      excluded[[v]] <- levels(x)[nlevels(x)]
      ok <- ok & x != levels(x)[nlevels(x)]
    }
  }
  eligible <- used[ok]
  set_fixture_seed(spec$predict_seed)
  a <- eligible[sample.int(length(eligible), n_new)]
  b <- eligible[sample.int(length(eligible), n_new)]
  newdata <- data[a, , drop = FALSE]
  newdata[parts$partitioning] <- data[b, parts$partitioning, drop = FALSE]
  rownames(newdata) <- NULL
  list(newdata = newdata, a = a, b = b, excluded = excluded)
}

formula_predict <- function(spec, tree, fx, ds) {
  nd <- formula_newdata(spec, tree, ds)
  newdata <- nd$newdata
  routes <- route_rows(tree, newdata)
  node <- vapply(routes, function(r) r$node, 0L)
  if (anyNA(node)) stop(spec$name, ": a new row is routed at random")
  node_r <- as.integer(predict(tree, newdata = newdata, type = "node"))
  if (!identical(node_r, node)) stop(spec$name, ": predict(type = \"node\") differs from the routing")
  resp <- as.numeric(predict(tree, newdata = newdata, type = "response"))
  # Check against the leaf coefficients written out: the model matrix of the
  # new rows with the model frame's levels, times the leaf's coefficients.
  rhs <- formula(Formula::Formula(tree$info$formula), lhs = 0L, rhs = 1L)
  mt <- stats::terms(rhs)
  xm <- stats::model.matrix(mt, stats::model.frame(mt, newdata, xlev = .getXlevels(mt, tree$data)))
  cf <- lapply(fx$nodes, function(n) stats::setNames(as.numeric(n$coefficients), as.character(n$coefficient_names)))
  names(cf) <- vapply(fx$nodes, function(n) as.character(n$id), "")
  eta <- vapply(seq_len(nrow(newdata)), function(i) {
    b <- cf[[as.character(node[i])]]
    if (!identical(names(b), colnames(xm))) stop(spec$name, ": model matrix columns differ from the coefficients")
    sum(xm[i, ] * b)
  }, 0)
  linkinv <- if (spec$fun == "lmtree") identity else eval(parse(text = spec$family))$linkinv
  check <- rel_diff_na(resp, linkinv(eta))
  if (check > 1e-12) stop(spec$name, ": predictions differ from the leaf coefficients written out")
  present <- lapply(Filter(is.factor, newdata), function(v) arr(sort(unique(as.character(v)))))
  list(
    note = "New rows: the regressors, response and weights of training row regressor_rows[i], the partitioning variables of training row partitioning_rows[i] (0-based indices into the data CSV), drawn from the rows the tree used that avoid excluded_levels. Factor columns keep the data's declared levels; levels_present lists the levels that occur. Every row is routed deterministically.",
    seed = spec$predict_seed,
    rng_kind = FIXTURE_RNG_KIND,
    n = nrow(newdata),
    regressor_rows = arr(nd$a - 1L),
    partitioning_rows = arr(nd$b - 1L),
    excluded_levels = named_list(nd$excluded),
    levels_present = named_list(present),
    columns = arr(names(newdata)),
    newdata = lapply(newdata, formula_json_column),
    r_calls = list(node = "predict(tree, newdata = newdata, type = \"node\")",
      response = "predict(tree, newdata = newdata, type = \"response\")"),
    expected = list(node = arr(node), response = arr(resp)),
    checks = list(response_vs_leaf_coefficients_max_rel = check)
  )
}

formula_tree_fixture <- function(spec, ds, instrumented) {
  common <- list(case = spec$case, r_outcome = spec$outcome)
  if (spec$outcome == "error") {
    data_env <- new.env(parent = globalenv())
    assign(spec$dataset, ds$data, envir = data_env)
    call_text <- tree_call_text(spec)
    warns <- character()
    err <- withCallingHandlers(
      tryCatch({
        eval(parse(text = call_text)[[1L]], data_env)
        NULL
      }, error = function(e) e),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    if (is.null(err)) stop(spec$name, ": expected R to stop")
    return(c(list(
      fixture = spec$name,
      description = spec$description,
      generator = "r/fixtures/formula.R",
      r_call = call_text,
      versions = fixture_versions(),
      dataset = spec$dataset,
      data_csv = file.path("data", paste0(spec$dataset, ".csv")),
      function_name = spec$fun,
      formula = spec$formula,
      family = NULL,
      weights = spec$weights,
      offset = spec$offset,
      control = control_json(spec$control)
    ), common, list(
      error = list(message = conditionMessage(err), call = deparse_value(conditionCall(err))),
      warnings = arr(unique(warns)),
      oracle_fixture = file.path("formula_trees", paste0(spec$oracle, ".json"))
    )))
  }
  fx <- tree_fixture(spec, ds, instrumented)
  fx$generator <- "r/fixtures/formula.R"
  data_env <- new.env(parent = globalenv())
  assign(spec$dataset, ds$data, envir = data_env)
  tree <- run_tree(tree_call_text(spec), data_env, NULL)$tree
  if (!identical(as.integer(tree$fitted[["(fitted)"]]), as.integer(fx$fitted_node))) {
    stop(spec$name, ": regrown tree differs from the fixture")
  }
  fx <- append(fx, c(common, list(expanded_formula = deparse_value(tree$info$formula))),
    after = match("formula", names(fx)))
  fx <- append(fx, model_frame_info(tree, ds), after = match("nobs", names(fx)))
  fx$print <- arr(utils::capture.output(print(tree)))
  fx$predict <- formula_predict(spec, tree, fx, ds)
  fx
}

# What the fixtures were made to show; stops generation otherwise.
check_formula_trees <- function(index) {
  fx <- lapply(index, `[[`, "fixture")
  names(fx) <- vapply(fx, `[[`, "", "fixture")
  split_vars <- function(f) vapply(Filter(function(nd) !isTRUE(nd$terminal), f$nodes),
    function(nd) nd$split$variable, "")
  expect_names <- list(
    sim_formula_factor_lmtree_factor = c("(Intercept)", "x", "fb", "fa", "fc"),
    sim_formula_factor_glmtree_binomial_factor = c("(Intercept)", "x", "fb", "fa", "fc"),
    sim_formula_factor_lmtree_noint = c("fd", "fb", "fa", "fc"),
    sim_formula_logical_lmtree_logical = c("(Intercept)", "x", "bTRUE"),
    sim_formula_logical_lmtree_logical_noint = c("bFALSE", "bTRUE", "x"),
    sim_formula_factor_lmtree_onepart = "(Intercept)",
    sim_formula_factor_glmtree_poisson_onepart = "(Intercept)")
  for (nm in names(expect_names)) {
    if (!identical(as.character(fx[[nm]]$coefficient_names), expect_names[[nm]])) {
      stop(nm, ": coefficient names ", paste(fx[[nm]]$coefficient_names, collapse = ", "))
    }
  }
  for (nm in c("sim_formula_factor_lmtree_factor", "sim_formula_factor_glmtree_binomial_factor",
    "sim_formula_factor_lmtree_noint", "sim_formula_logical_lmtree_logical",
    "sim_formula_logical_lmtree_logical_noint", "sim_formula_factor_lmtree_onepart",
    "sim_formula_factor_glmtree_poisson_onepart")) {
    if (!identical(unique(split_vars(fx[[nm]])), "z1")) stop(nm, ": does not split on z1 only")
  }
  if (!identical(split_vars(fx$sim_formula_logical_numeric_lmtree_split)[1L], "zb")) {
    stop("sim_formula_logical_numeric_lmtree_split: the root does not split on zb")
  }
  if (!identical(sort(unique(split_vars(fx$sim_formula_ordered_lmtree))), c("zn", "zo"))) {
    stop("sim_formula_ordered_lmtree: does not split on zo and zn")
  }
  na <- fx$sim_formula_na_lmtree_weights
  if (!all(c("z1", "zf") %in% split_vars(na))) stop("sim_formula_na_lmtree_weights: does not split on z1 and zf")
  if (!identical(as.character(na$model_frame$dropped_levels$f), "e") ||
      !identical(as.character(na$model_frame$dropped_levels$zf), "q")) {
    stop("sim_formula_na_lmtree_weights: the levels e and q are not dropped")
  }
  err <- fx$sim_formula_logical_lmtree_logical_split
  if (!identical(err$error$message, "variable 2 is not integer")) {
    stop("sim_formula_logical_lmtree_logical_split: unexpected error ", err$error$message)
  }
  invisible(TRUE)
}
