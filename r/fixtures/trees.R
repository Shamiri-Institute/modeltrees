# Tree fixtures: lmtree() and glmtree() on the datasets in datasets.R.

# Every mob_control() argument except the minsplit/minbucket aliases. Each
# fixture's call lists all of them explicitly.
MOB_CONTROL_DEFAULTS <- list(
  alpha = 0.05, bonferroni = TRUE, minsize = NULL, maxdepth = Inf,
  mtry = Inf, trim = 0.1, breakties = FALSE, parm = NULL, dfsplit = TRUE,
  prune = NULL, restart = TRUE, verbose = FALSE, caseweights = TRUE,
  ytype = "vector", xtype = "matrix",
  # Store scores and model objects in every node so the fixtures can expose
  # them. These two options only control storage.
  terminal = c("estfun", "object"), inner = c("estfun", "object"),
  model = TRUE, numsplit = "left", catsplit = "binary", vcov = "opg",
  ordinal = "chisq", nrep = 10000, applyfun = NULL, cores = NULL
)

GLM_CONTROL_DEFAULTS <- list(epsilon = 1e-8, maxit = 25)

tree_spec <- function(name, dataset, fun, formula, description, expect,
                      family = NULL, weights = NULL, offset = NULL,
                      control = list()) {
  unknown <- setdiff(names(control), names(MOB_CONTROL_DEFAULTS))
  if (length(unknown)) stop("Unknown control settings: ", paste(unknown, collapse = ", "))
  ctrl <- MOB_CONTROL_DEFAULTS
  ctrl[names(control)] <- control
  list(name = name, dataset = dataset, fun = fun, formula = formula,
    description = description, expect = expect, family = family,
    weights = weights, offset = offset, control = ctrl)
}

all_tree_specs <- function() {
  zg <- "z1 + z2 + z3 + z4 + z5"
  zb <- "z1 + z2 + z3 + z4"
  sd <- "diabetes ~ glucose | pressure + triceps + insulin + mass + age"
  list(
    tree_spec("synth_diabetes_glmtree_binomial", "synth_diabetes", "glmtree", sd,
      "The mob vignette's glmtree example with default control settings.",
      "several", family = "binomial(link = \"logit\")"),
    tree_spec("synth_diabetes_glmtree_binomial_bic", "synth_diabetes", "glmtree", sd,
      "The vignette's post-pruning strategy: large alpha, maxdepth 4, BIC pruning.",
      "several", family = "binomial(link = \"logit\")",
      control = list(minsize = 50L, maxdepth = 4L, alpha = 0.9, prune = "BIC")),
    tree_spec("sim_gaussian_lmtree_nosplit", "sim_gaussian", "lmtree",
      paste("y_null ~ x |", zg), "No planted instability.", "none",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_lmtree_onesplit", "sim_gaussian", "lmtree",
      paste("y_one ~ x |", zg), "One planted split at z1 = 5.", "one",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_lmtree_multisplit", "sim_gaussian", "lmtree",
      paste("y_multi ~ x |", zg),
      "Planted splits on numeric z1, ordered z3 and nominal z2.", "several",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_glmtree_gaussian_multisplit", "sim_gaussian", "glmtree",
      paste("y_multi ~ x |", zg), "As the lmtree fixture, fitted by glmtree.",
      "several", family = "gaussian(link = \"identity\")",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_lmtree_intercept_only", "sim_gaussian", "lmtree",
      paste("y_one ~ 1 |", zg), "Constant fits, one parameter per node.", "one",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_lmtree_weights", "sim_gaussian", "lmtree",
      paste("y_multi ~ x |", zg), "Positive integer case weights.", "several",
      weights = "w", control = list(minsize = 30L)),
    tree_spec("sim_gaussian_glmtree_gaussian_weights", "sim_gaussian", "glmtree",
      paste("y_multi ~ x |", zg),
      "Positive integer case weights; exercises the case-weight dispersion in glmfit's scores.",
      "several", family = "gaussian(link = \"identity\")", weights = "w",
      control = list(minsize = 30L)),
    tree_spec("sim_gaussian_lmtree_zero_weights", "sim_gaussian", "lmtree",
      paste("y_multi ~ x |", zg),
      "Edge case: case weights with zeros. partykit 1.3-0 fails the root instability test and returns a single node.",
      "none", weights = "w0", control = list(minsize = 30L)),
    tree_spec("sim_binomial_glmtree_nosplit", "sim_binomial", "glmtree",
      paste("y_null ~ x |", zb), "No planted instability.", "none",
      family = "binomial(link = \"logit\")", control = list(minsize = 40L)),
    tree_spec("sim_binomial_glmtree_onesplit", "sim_binomial", "glmtree",
      paste("y_one ~ x |", zb), "One planted split at z1 = 0.", "one",
      family = "binomial(link = \"logit\")", control = list(minsize = 40L)),
    tree_spec("sim_binomial_glmtree_multisplit", "sim_binomial", "glmtree",
      paste("y_multi ~ x |", zb), "Planted splits on nominal z2 and numeric z1.",
      "several", family = "binomial(link = \"logit\")", control = list(minsize = 40L)),
    tree_spec("sim_poisson_glmtree_nosplit", "sim_poisson", "glmtree",
      paste("y_null ~ x |", zb), "No planted instability, with offset.", "none",
      family = "poisson(link = \"log\")", offset = "log(exposure)",
      control = list(minsize = 40L)),
    tree_spec("sim_poisson_glmtree_multisplit", "sim_poisson", "glmtree",
      paste("y_multi ~ x |", zb), "Planted splits on ordered z3 and numeric z1, with offset.",
      "several", family = "poisson(link = \"log\")", offset = "log(exposure)",
      control = list(minsize = 40L)),
    tree_spec("sim_poisson_glmtree_weights", "sim_poisson", "glmtree",
      paste("y_multi ~ x |", zb), "Positive integer case weights and an offset.",
      "several", family = "poisson(link = \"log\")", offset = "log(exposure)",
      weights = "w", control = list(minsize = 40L))
  )
}

# ---------------------------------------------------------------------------
# Building and evaluating the call

deparse_value <- function(v) paste(deparse(v, width.cutoff = 500L), collapse = "")

tree_call_text <- function(spec) {
  args <- c(
    sprintf("formula = %s", spec$formula),
    sprintf("data = %s", spec$dataset),
    if (!is.null(spec$family)) sprintf("family = %s", spec$family),
    if (!is.null(spec$weights)) sprintf("weights = %s", spec$weights),
    if (!is.null(spec$offset)) sprintf("offset = %s", spec$offset),
    if (spec$fun == "glmtree") {
      sprintf("%s = %s", names(GLM_CONTROL_DEFAULTS), vapply(GLM_CONTROL_DEFAULTS, deparse_value, ""))
    },
    sprintf("%s = %s", names(spec$control), vapply(spec$control, deparse_value, ""))
  )
  sprintf("partykit::%s(%s)", spec$fun, paste(args, collapse = ", "))
}

# JSON form of a control value: Inf becomes the string "Inf", NULL becomes null.
control_json <- function(ctrl) {
  lapply(ctrl, function(v) {
    if (is.null(v)) return(NULL)
    if (is.numeric(v) && length(v) == 1L && is.infinite(v)) return("Inf")
    if (length(v) > 1L) return(I(v))
    v
  })
}

run_tree <- function(call_text, data_env, instrumented) {
  warnings <- character()
  tree <- withCallingHandlers(
    {
      expr <- parse(text = call_text)[[1L]]
      if (is.null(instrumented)) {
        eval(expr, data_env)
      } else {
        with_mob_partynode(instrumented, eval(expr, data_env))
      }
    },
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(tree = tree, warnings = unique(warnings))
}

# ---------------------------------------------------------------------------
# Extraction

arr <- function(x) I(x)

node_rows <- function(tree, id) {
  fitted <- tree$fitted[["(fitted)"]]
  which(fitted %in% partykit::nodeids(tree, from = id, terminal = TRUE)) - 1L
}

describe_split <- function(split, z_names, mf) {
  var_name <- names(mf)[split$varid]
  z <- mf[[var_name]]
  out <- list(
    variable = var_name,
    variable_index = match(var_name, z_names) - 1L,
    partykit_varid = split$varid,
    right = split$right
  )
  if (is.null(split$index)) {
    if (is.factor(z)) {
      out$kind <- "ordered"
      out$breaks <- split$breaks
      out$break_level <- levels(z)[split$breaks]
      out$levels <- arr(levels(z))
      out$left_levels <- arr(levels(z)[seq_len(split$breaks)])
      out$right_levels <- arr(levels(z)[-seq_len(split$breaks)])
    } else {
      out$kind <- "numeric"
      out$breaks <- split$breaks
    }
  } else {
    out$kind <- "nominal"
    out$index <- arr(split$index)
    out$levels <- arr(levels(z))
    out$left_levels <- arr(levels(z)[which(split$index == 1L)])
    out$right_levels <- arr(levels(z)[which(split$index == 2L)])
    out$absent_levels <- arr(levels(z)[is.na(split$index)])
  }
  out
}

describe_test <- function(info, z_names) {
  if (is.null(info$test)) {
    return(list(status = "not_run"))
  }
  if (is.null(info$.fx_raw_test)) {
    return(list(status = "failed"))
  }
  raw <- info$.fx_raw_test
  best <- raw$best
  list(
    status = "ok",
    variables = arr(z_names),
    tested = arr(raw$.fx_tested),
    statistic = arr(finite_or_na(info$test["statistic", ])),
    p_value_adjusted = arr(finite_or_na(info$test["p.value", ])),
    p_value_raw = arr(finite_or_na(raw$pval)),
    log_p_value_raw = arr(finite_or_na(raw$.fx_logpval)),
    best = if (is.na(best)) NULL else unname(best) - 1L,
    best_variable = if (is.na(best)) NULL else names(best),
    nobs = raw$.fx_nobs,
    k = raw$.fx_k,
    trim_from = raw$.fx_from,
    trim_to = raw$.fx_to,
    lambda = raw$.fx_lambda
  )
}

stop_reason <- function(nd, info, alpha) {
  if (!is.null(nd$kids)) return("split")
  if (is.null(info$test)) {
    return(if (info$nobs < 2 * info$.fx_minsize) "minsize" else "maxdepth")
  }
  if (is.null(info$.fx_raw_test)) return("test_failed")
  if (is.na(info$p.value) || info$p.value > alpha) return("not_significant")
  ss <- info$.fx_split_search
  if (!is.null(ss) && all(!is.finite(ss$objective))) return("no_admissible_split")
  "pruned"
}

extract_nodes <- function(tree, alpha) {
  nodes <- as.list(tree$node)
  mf <- tree$data
  z_names <- all.vars(tree$info$terms$partitioning)
  parent <- integer(length(nodes))
  depth <- integer(length(nodes))
  depth[1L] <- 1L
  for (nd in nodes) {
    for (kid in nd$kids) {
      parent[kid] <- nd$id
      depth[kid] <- depth[nd$id] + 1L
    }
  }
  lapply(nodes, function(nd) {
    info <- nd$info
    cf <- info$coefficients
    ef <- info$estfun
    ss <- info$.fx_split_search
    list(
      id = nd$id,
      parent = if (parent[nd$id] == 0L) NULL else parent[nd$id],
      depth = depth[nd$id],
      kids = arr(if (is.null(nd$kids)) integer() else as.integer(nd$kids)),
      terminal = is.null(nd$kids),
      stop_reason = stop_reason(nd, info, alpha),
      nobs = info$nobs,
      n_rows = length(node_rows(tree, nd$id)),
      minsize = info$.fx_minsize,
      rows = arr(node_rows(tree, nd$id)),
      coefficient_names = arr(names(cf)),
      coefficients = arr(finite_or_na(cf)),
      objfun = info$objfun,
      p_value = if (is.null(info$p.value)) NULL else info$p.value,
      test = describe_test(info, z_names),
      split = if (is.null(nd$split)) NULL else describe_split(nd$split, z_names, mf),
      split_search = if (is.null(ss)) NULL else list(
        variable = ss$variable,
        minsize = ss$minsize,
        candidates = arr(if (is.numeric(ss$candidates)) ss$candidates else as.character(ss$candidates)),
        objective = arr(finite_or_na(ss$objective))
      ),
      estfun_names = arr(colnames(ef)),
      estfun = unname(ef)
    )
  })
}

# How close the fixture sits to a decision boundary. Small values mean a
# Python port could legitimately disagree with R through rounding.
decision_margins <- function(nodes, alpha) {
  p_gap <- unlist(lapply(nodes, function(nd) {
    if (nd$test$status == "ok" && !is.null(nd$p_value)) abs(nd$p_value - alpha) else NULL
  }))
  split_gap <- unlist(lapply(nodes, function(nd) {
    if (is.null(nd$split_search)) return(NULL)
    o <- sort(as.numeric(nd$split_search$objective))
    if (length(o) < 2L) return(NULL)
    (o[2L] - o[1L]) / max(1, abs(o[1L]))
  }))
  list(
    min_abs_p_value_minus_alpha = if (length(p_gap)) min(p_gap) else NULL,
    min_relative_split_objective_gap = if (length(split_gap)) min(split_gap) else NULL
  )
}

tree_fixture <- function(spec, ds, instrumented) {
  data_env <- new.env(parent = globalenv())
  assign(spec$dataset, ds$data, envir = data_env)
  call_text <- tree_call_text(spec)

  plain <- run_tree(call_text, data_env, NULL)
  inst <- run_tree(call_text, data_env, instrumented)
  if (!identical(comparable_nodes(as.list(plain$tree$node)),
                 comparable_nodes(strip_fx(inst$tree))) ||
      !identical(plain$tree$fitted, inst$tree$fitted)) {
    stop("Instrumented mob_partynode changed the tree for ", spec$name)
  }
  tree <- inst$tree
  stopifnot(nrow(tree$data) == nrow(ds$data))

  nodes <- extract_nodes(tree, spec$control$alpha)
  n_terminal <- length(partykit::nodeids(tree, terminal = TRUE))
  got <- if (n_terminal == 1L) "none" else if (n_terminal == 2L) "one" else "several"
  if (got != spec$expect) {
    stop(sprintf("%s: expected %s split(s), got %d terminal nodes", spec$name, spec$expect, n_terminal))
  }

  y <- tree$data[[1L]]
  ll <- stats::logLik(tree)
  family <- if (spec$fun == "lmtree") NULL else {
    f <- eval(parse(text = spec$family))
    list(family = f$family, link = f$link)
  }

  list(
    fixture = spec$name,
    description = spec$description,
    generator = "r/fixtures/trees.R",
    r_call = call_text,
    versions = fixture_versions(),
    dataset = spec$dataset,
    data_csv = file.path("data", paste0(spec$dataset, ".csv")),
    function_name = spec$fun,
    formula = spec$formula,
    response = names(tree$data)[1L],
    response_levels = if (is.factor(y)) arr(levels(y)) else NULL,
    response_encoding = if (is.factor(y)) "binomial: 0 for the first level, 1 otherwise (glm.fit family$initialize)" else NULL,
    regressors = arr(all.vars(formula(Formula::Formula(tree$info$formula), lhs = 0L, rhs = 1L))),
    partitioning = arr(all.vars(tree$info$terms$partitioning)),
    coefficient_names = nodes[[1L]]$coefficient_names,
    family = family,
    weights = spec$weights,
    offset = spec$offset,
    control = control_json(spec$control),
    glm_control = if (spec$fun == "glmtree") GLM_CONTROL_DEFAULTS else NULL,
    root_minsize = nodes[[1L]]$minsize,
    n = nrow(tree$data),
    nobs = stats::nobs(tree),
    warnings = arr(inst$warnings),
    summary = list(
      n_nodes = length(nodes),
      n_inner = length(nodes) - n_terminal,
      n_terminal = n_terminal,
      depth = max(vapply(nodes, function(nd) nd$depth, 0L)),
      objfun = sum(vapply(Filter(function(nd) nd$terminal, nodes), function(nd) nd$objfun, 0)),
      loglik = as.numeric(ll),
      loglik_df = attr(ll, "df")
    ),
    decision_margins = decision_margins(nodes, spec$control$alpha),
    fitted_node = arr(as.integer(tree$fitted[["(fitted)"]])),
    nodes = nodes
  )
}
