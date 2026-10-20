# predict() on new data for three existing tree fixtures, one fixture per tree
# in tests/fixtures/predict/. Sourced by r/generate_all.R after trees.R.
#
# partykit routes a row at random (kidids_node, node.R:189-194) when the split
# variable is missing or its level has no kid at the node (index NA). Such
# rows are marked "random" and get no expected values. The marking comes from
# partykit's own kidids_split without the random fallback, and is checked by
# predicting with several seeds: every "deterministic" row must get the same
# node each time.

PREDICT_TREES <- c("sim_gaussian_lmtree_weights", "sim_binomial_glmtree_multisplit",
  "sim_poisson_glmtree_multisplit")

PREDICT_SEEDS <- 1:25

# The kid of node `nd` for one row of newdata, NA where partykit would draw
# at random, with the reason.
route_rows <- function(tree, newdata) {
  vmatch <- match(names(tree$data), names(newdata))
  z_names <- names(tree$data)
  lapply(seq_len(nrow(newdata)), function(i) {
    nd <- partykit::node_party(tree)
    path <- integer()
    while (!partykit::is.terminal(nd)) {
      sp <- partykit::split_node(nd)
      k <- partykit:::kidids_split(sp, newdata[i, , drop = FALSE], vmatch)
      path <- c(path, nd$id)
      if (is.na(k)) {
        v <- z_names[sp$varid]
        val <- newdata[[v]][i]
        reason <- if (is.na(val)) {
          sprintf("%s is missing at node %d", v, nd$id)
        } else if (is.factor(val)) {
          sprintf("level %s of %s is absent from node %d (split index NA)", as.character(val), v, nd$id)
        } else {
          sprintf("%s = %s is outside .bincode's intervals at node %d", v, format(val), nd$id)
        }
        return(list(node = NA_integer_, path = path, reason = reason))
      }
      nd <- nd$kids[[k]]
    }
    list(node = nd$id, path = c(path, nd$id), reason = NULL)
  })
}

# Split variables on the path from the root to node `id`.
path_variables <- function(tree, id) {
  z_names <- names(tree$data)
  out <- character()
  nd <- partykit::node_party(tree)
  while (nd$id != id) {
    sp <- partykit::split_node(nd)
    out <- c(out, z_names[sp$varid])
    kids <- vapply(nd$kids, function(k) k$id, 0L)
    nd <- nd$kids[[max(which(kids <= id))]]
  }
  unique(out)
}

# New rows built from training rows: each case starts from the first training
# row of a node (so it reaches that node) and changes some columns.
predict_cases <- function(tree, tfx, data, offset_var) {
  z_names <- as.character(tfx$partitioning)
  nodes <- tfx$nodes
  cases <- list()
  add <- function(case, src, changes, note) {
    row <- data[src + 1L, , drop = FALSE]
    for (nm in names(changes)) {
      v <- changes[[nm]]
      row[[nm]] <- if (is.factor(data[[nm]])) {
        factor(if (is.na(v)) NA_character_ else v, levels = levels(data[[nm]]),
          ordered = is.ordered(data[[nm]]))
      } else as.numeric(v)
    }
    cases[[length(cases) + 1L]] <<- list(case = case, target_node = target, source_row = src,
      changes = changes, note = note, row = row)
  }
  for (nd in nodes) {
    target <- nd$id
    src <- nd$rows[[1L]]
    sp <- nd$split
    if (!is.null(sp)) {
      v <- sp$variable
      if (sp$kind == "numeric") {
        b <- sp$breaks
        add("at_break", src, stats::setNames(list(b), v),
          sprintf("%s exactly at the break of node %d (right = TRUE: goes to kid 1)", v, nd$id))
        add("above_break", src, stats::setNames(list(b + abs(b) * 2^-52), v),
          sprintf("%s one step above the break of node %d", v, nd$id))
        add("minus_inf", src, stats::setNames(list(-Inf), v), sprintf("%s = -Inf at node %d", v, nd$id))
        add("plus_inf", src, stats::setNames(list(Inf), v), sprintf("%s = Inf at node %d", v, nd$id))
      } else if (sp$kind == "ordered") {
        lev <- as.character(sp$levels)
        add("at_break", src, stats::setNames(list(sp$break_level), v),
          sprintf("%s at the break level %s of node %d", v, sp$break_level, nd$id))
        add("above_break", src, stats::setNames(list(lev[sp$breaks + 1L]), v),
          sprintf("%s at the level above the break of node %d", v, nd$id))
        # Ordered levels absent from the node still route by the break.
        present <- unique(as.character(data[[v]][nd$rows + 1L]))
        for (l in setdiff(lev, present)) {
          add("absent_level", src, stats::setNames(list(l), v),
            sprintf("ordered level %s, absent from node %d", l, nd$id))
        }
      } else {
        for (l in as.character(sp$absent_levels)) {
          add("absent_level", src, stats::setNames(list(l), v),
            sprintf("level %s, absent from node %d", l, nd$id))
        }
      }
      add("missing_on_path", src, stats::setNames(list(NA), v),
        sprintf("%s missing at node %d, where it is the split variable", v, nd$id))
    } else {
      on_path <- path_variables(tree, nd$id)
      off <- setdiff(z_names, on_path)
      add("missing_off_path", src, stats::setNames(as.list(rep(NA, length(off))), off),
        sprintf("leaf %d's row with every partitioning variable off its path missing (%s)",
          nd$id, paste(off, collapse = ", ")))
      add("missing_regressor", src, list(x = NA),
        sprintf("leaf %d's row with the regressor x missing", nd$id))
      if (!is.null(offset_var)) {
        add("missing_offset", src, stats::setNames(list(NA), offset_var),
          sprintf("leaf %d's row with %s (the offset variable) missing", nd$id, offset_var))
      }
      add("training_row", src, list(), sprintf("leaf %d's first training row, unchanged", nd$id))
    }
  }
  cases
}

# Columns of newdata for JSON: factors as labels, numbers with null for NA and
# the strings "Inf" and "-Inf".
json_column <- function(v) {
  if (is.factor(v)) return(arr(ifelse(is.na(v), NA_character_, as.character(v))))
  lapply(as.numeric(v), function(x) {
    if (is.na(x)) NA else if (x == Inf) "Inf" else if (x == -Inf) "-Inf" else x
  })
}

na_null <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA_real_
  arr(x)
}

predict_fixture <- function(spec, tfx, ds) {
  data_env <- new.env(parent = globalenv())
  assign(spec$dataset, ds$data, envir = data_env)
  tree <- run_tree(tree_call_text(spec), data_env, NULL)$tree
  if (!identical(as.integer(tree$fitted[["(fitted)"]]), as.integer(tfx$fitted_node))) {
    stop(spec$name, ": regrown tree differs from the fixture")
  }
  offset_var <- if (is.null(spec$offset)) NULL else all.vars(parse(text = spec$offset)[[1L]])
  cases <- predict_cases(tree, tfx, ds$data, offset_var)
  newdata <- do.call(rbind, lapply(cases, `[[`, "row"))
  rownames(newdata) <- NULL

  routes <- route_rows(tree, newdata)
  node_det <- vapply(routes, function(r) r$node, 0L)
  det <- !is.na(node_det)

  # Seeds: deterministic rows never move, and random rows do move under
  # some seed (otherwise the marking would be suspect).
  by_seed <- vapply(PREDICT_SEEDS, function(s) {
    set_fixture_seed(s)
    as.integer(predict(tree, newdata = newdata, type = "node"))
  }, integer(nrow(newdata)))
  if (any(by_seed[det, , drop = FALSE] != node_det[det])) stop(spec$name, ": a deterministic row moved")
  moved <- apply(by_seed, 1L, function(r) length(unique(r)) > 1L)
  if (any(!det & !moved)) stop(spec$name, ": a row marked random never moved over the seeds")

  nd_det <- newdata[det, , drop = FALSE]
  pred <- function(type) {
    tryCatch(as.numeric(predict(tree, newdata = nd_det, type = type)), error = function(e) e)
  }
  node_r <- as.integer(predict(tree, newdata = nd_det, type = "node"))
  if (!identical(node_r, node_det[det])) stop(spec$name, ": predict(type = \"node\") differs from the routing")
  resp <- pred("response")
  link <- pred("link")
  fill <- function(p) {
    if (inherits(p, "error")) return(NULL)
    out <- rep(NA_real_, nrow(newdata))
    out[det] <- p
    na_null(out)
  }

  # Check the predictions against the leaf coefficients written out:
  # X beta (+ offset), and the inverse link.
  cf <- lapply(tfx$nodes, function(nd) as.numeric(nd$coefficients))
  names(cf) <- vapply(tfx$nodes, function(nd) as.character(nd$id), "")
  off <- if (is.null(spec$offset)) rep(0, nrow(nd_det)) else eval(parse(text = spec$offset)[[1L]], nd_det)
  eta <- vapply(seq_len(nrow(nd_det)), function(i) {
    b <- cf[[as.character(node_r[i])]]
    b[1L] + b[2L] * nd_det$x[i] + off[i]
  }, 0)
  linkinv <- if (spec$fun == "lmtree") identity else eval(parse(text = spec$family))$linkinv
  by_hand <- linkinv(eta)
  check_resp <- rel_diff_na(resp, by_hand)
  check_link <- if (inherits(link, "error")) NULL else rel_diff_na(link, eta)
  if (check_resp > 1e-12 || (!is.null(check_link) && check_link > 1e-12)) {
    stop(spec$name, ": predictions differ from the leaf coefficients written out")
  }

  # Calls R does not answer.
  failing <- list()
  if (inherits(link, "error")) {
    failing[[length(failing) + 1L]] <- list(
      r_call = "predict(tree, newdata = newdata[deterministic, ], type = \"link\")",
      error = conditionMessage(link),
      note = "predict.lmtree passes type to predict.lm, which has no \"link\" type")
  }

  # An unseen level (an extra factor level in newdata). partykit compares the
  # levels of the split variables only (predict.party, party.R:295-310): a
  # split variable with other levels sends it to model.frame(xlev = ...),
  # which stops; a partitioning variable used by no split is not checked.
  split_vars <- unique(vapply(Filter(function(nd) !is.null(nd$split), tfx$nodes),
    function(nd) nd$split$variable, ""))
  factor_vars <- names(Filter(is.factor, ds$data[as.character(tfx$partitioning)]))
  unseen_calls <- list()
  for (fz in c(intersect(factor_vars, split_vars)[1L], setdiff(factor_vars, split_vars)[1L])) {
    if (is.na(fz)) next
    one <- newdata[nrow(newdata), , drop = FALSE]
    one[[fz]] <- factor("zz_unseen", levels = c(levels(ds$data[[fz]]), "zz_unseen"),
      ordered = is.ordered(ds$data[[fz]]))
    r <- tryCatch(as.integer(predict(tree, newdata = one, type = "node")), error = function(e) e)
    used <- fz %in% split_vars
    if (used != inherits(r, "error")) stop(spec$name, ": unexpected outcome for an unseen level of ", fz)
    unseen_calls[[length(unseen_calls) + 1L]] <- list(
      r_call = sprintf("predict(tree, newdata = newdata[%d, ] with %s = \"zz_unseen\" added as an extra level, type = \"node\")",
        nrow(newdata), fz),
      variable = fz,
      split_variable = used,
      error = if (used) conditionMessage(r) else NULL,
      node = if (used) NULL else r,
      node_without_change = node_det[nrow(newdata)],
      note = if (used) "the variable is used in a split: its levels differ from the training levels, so partykit goes through model.frame(xlev = ...), which stops"
        else "the variable is used in no split, so partykit does not check its levels and routes the row as before")
  }

  list(
    fixture = paste0("predict_", spec$name),
    description = sprintf("predict() of trees/%s.json on new rows built from training rows: break values, absent levels, missing values on and off the path, a missing regressor%s.",
      spec$name, if (is.null(offset_var)) "" else " and a missing offset"),
    generator = "r/fixtures/predict.R",
    versions = fixture_versions(),
    tree_fixture = file.path("trees", paste0(spec$name, ".json")),
    r_call = tfx$r_call,
    data_csv = tfx$data_csv,
    function_name = spec$fun,
    offset = spec$offset,
    n = nrow(newdata),
    columns = arr(names(newdata)),
    newdata = lapply(newdata, json_column),
    cases = lapply(seq_along(cases), function(i) {
      c0 <- cases[[i]]
      list(row = i - 1L, case = c0$case, target_node = c0$target_node,
        reaches_target_node = c0$target_node %in% routes[[i]]$path,
        source_row = c0$source_row,
        changed = arr(as.character(names(c0$changes))), note = c0$note,
        routing = if (det[i]) "deterministic" else "random",
        random_reason = routes[[i]]$reason,
        path = arr(routes[[i]]$path))
    }),
    random_check = sprintf("predict(type = \"node\") under set.seed(s) for s in %d:%d: rows marked deterministic never move, rows marked random move at least once",
      min(PREDICT_SEEDS), max(PREDICT_SEEDS)),
    expected = list(
      note = "One value per newdata row; null for rows routed at random (left out of R's call, which predicts newdata[deterministic, ]) and for R's NA predictions (a missing regressor or offset).",
      node = arr(ifelse(det, node_det, NA_integer_)),
      response = fill(resp),
      link = fill(link)
    ),
    checks = list(
      response_vs_leaf_coefficients_max_rel = check_resp,
      link_vs_leaf_coefficients_max_rel = check_link
    ),
    failing_calls = failing,
    unseen_level_calls = unseen_calls
  )
}

# Relative difference over the entries finite in both; NA must match.
rel_diff_na <- function(a, b) {
  if (!identical(is.na(a), is.na(b))) return(Inf)
  ok <- !is.na(a)
  if (!any(ok)) return(0)
  max(abs(a[ok] - b[ok]) / pmax(1, abs(b[ok])))
}
