# Shared helpers for the fixture generators. Sourced by r/generate_all.R.

FIXTURE_RNG_KIND <- c("Mersenne-Twister", "Inversion", "Rejection")

# Package versions recorded in every fixture.
fixture_versions <- function() {
  pkgs <- c("partykit", "strucchange", "sandwich", "Formula", "jsonlite")
  c(
    list(R = paste(R.version$major, R.version$minor, sep = ".")),
    stats::setNames(lapply(pkgs, function(p) utils::packageDescription(p)$Version), pkgs)
  )
}

set_fixture_seed <- function(seed) {
  RNGkind(kind = FIXTURE_RNG_KIND[1L], normal.kind = FIXTURE_RNG_KIND[2L],
    sample.kind = FIXTURE_RNG_KIND[3L])
  set.seed(seed)
}

# ---------------------------------------------------------------------------
# Output

# Replace non-finite numbers with NULL-producing NA so they serialize as null.
# Callers document what null means for each field.
finite_or_na <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA_real_
  x
}

# 17 significant digits round-trip every double exactly. jsonlite's
# digits = NA keeps only 15 and does not.
write_fixture_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  txt <- jsonlite::toJSON(x, pretty = TRUE, auto_unbox = TRUE, digits = I(17),
    na = "null", null = "null")
  writeLines(txt, path, useBytes = TRUE)
  invisible(path)
}

# 17 significant digits always round-trip a double under a correctly rounded
# parser. R's own as.numeric() is not correctly rounded, so no round-trip check
# is made here.
format_double <- function(x) {
  sprintf("%.17g", x)
}

# Describe each column so Python can rebuild identical types.
column_schema <- function(df) {
  lapply(names(df), function(nm) {
    v <- df[[nm]]
    if (is.ordered(v)) {
      list(name = nm, type = "ordered", levels = levels(v))
    } else if (is.factor(v)) {
      list(name = nm, type = "factor", levels = levels(v))
    } else if (is.integer(v)) {
      list(name = nm, type = "integer")
    } else if (is.numeric(v)) {
      list(name = nm, type = "double")
    } else {
      stop("Unsupported column type for ", nm)
    }
  })
}

# Write a dataset as CSV plus a JSON schema. Doubles use round-trip formatting,
# factors are written as their labels. No row names, no quoting needed because
# labels are plain identifiers (checked).
write_dataset <- function(ds, data_dir) {
  df <- ds$data
  stopifnot(!anyNA(df))
  cols <- lapply(df, function(v) {
    if (is.factor(v)) {
      lab <- as.character(v)
      stopifnot(all(grepl("^[A-Za-z0-9_.]+$", lab)))
      lab
    } else if (is.integer(v)) {
      as.character(v)
    } else {
      format_double(v)
    }
  })
  lines <- c(paste(names(df), collapse = ","), do.call(paste, c(cols, sep = ",")))
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  csv_path <- file.path(data_dir, paste0(ds$name, ".csv"))
  writeLines(lines, csv_path, useBytes = TRUE)

  schema <- list(
    dataset = ds$name,
    description = ds$description,
    source = ds$source,
    seed = ds$seed,
    rng_kind = if (is.null(ds$seed)) NULL else FIXTURE_RNG_KIND,
    n = nrow(df),
    csv = paste0(ds$name, ".csv"),
    columns = column_schema(df),
    generator_code = ds$code
  )
  if (!is.null(ds$removed_rows)) {
    schema$source_dataset <- ds$source_dataset
    schema$removed_rows <- I(ds$removed_rows)
  }
  write_fixture_json(schema, file.path(data_dir, paste0(ds$name, ".schema.json")))
  invisible(csv_path)
}

# ---------------------------------------------------------------------------
# Instrumented copy of partykit:::mob_partynode
#
# partykit keeps only Bonferroni-adjusted p-values and discards the objective
# of every split candidate. The copy below is the deparsed partykit function
# with four insertions that store those values in the node info under names
# starting with ".fx_". Nothing else changes. generate_all.R grows every tree
# with both the original and the instrumented function and requires identical
# results apart from the .fx_ fields.

insert_after <- function(src, pattern, new_lines) {
  hit <- which(grepl(pattern, src, fixed = TRUE))
  if (length(hit) != 1L) stop("Patch anchor not unique: ", pattern)
  append(src, new_lines, after = hit)
}

insert_before <- function(src, pattern, new_lines) {
  hit <- which(grepl(pattern, src, fixed = TRUE))
  if (length(hit) != 1L) stop("Patch anchor not unique: ", pattern)
  append(src, new_lines, after = hit - 1L)
}

build_instrumented_mob_partynode <- function() {
  original <- partykit:::mob_partynode
  src <- deparse(original, width.cutoff = 500L)

  # 1. Keep log p-values and trimming quantities from the fluctuation tests.
  src <- insert_after(src, "rval <- list(pval = exp(pval), stat = stat, best = best)",
    "rval$.fx_logpval <- pval; rval$.fx_from <- from; rval$.fx_to <- to; rval$.fx_lambda <- lambda; rval$.fx_nobs <- nobs; rval$.fx_k <- k; rval$.fx_tested <- seq_len(m) %in% mtest")

  # 1b. Record the minsize in force at each node, including the default that
  # partykit computes at the root and never stores.
  src <- insert_after(src, "minsize <<- as.integer(ceiling(10L * length(mod$coefficients)/NCOL(y)))",
    "        mod$.fx_minsize <- minsize")

  # 2. Store the raw test results before the multiplicity adjustment.
  src <- insert_after(src, "if (!is.null(test) && !inherits(test, \"try-error\")) {",
    "mod$.fx_raw_test <- test")

  # 3. Return split candidates and their objectives from the split search.
  src <- insert_before(src, "        return(split)",
    "        if (exists(\"dev\", inherits = FALSE)) { split$.fx_dev <- dev; split$.fx_candidates <- if (is.numeric(zselect)) uz else rownames(al) }")

  # 4. Keep them in the node info.
  src <- insert_after(src, "sp <- mob_grow_findsplit(y, x, zselect, weights, offset, cluster, ...)",
    "mod$.fx_split_search <- list(variable = names(z)[best], candidates = sp$.fx_candidates, objective = sp$.fx_dev, minsize = minsize)")

  patched <- eval(parse(text = src))
  environment(patched) <- environment(original)
  patched
}

# Run expr with partykit:::mob_partynode temporarily replaced by fn.
with_mob_partynode <- function(fn, expr) {
  ns <- asNamespace("partykit")
  original <- get("mob_partynode", envir = ns)
  utils::assignInNamespace("mob_partynode", fn, ns = "partykit")
  on.exit(utils::assignInNamespace("mob_partynode", original, ns = "partykit"))
  force(expr)
}

strip_fx <- function(tree) {
  nodes <- as.list(tree$node)
  lapply(nodes, function(nd) {
    if (!is.null(nd$info)) nd$info <- nd$info[!startsWith(names(nd$info), ".fx_")]
    nd
  })
}

# Drop the fitted model objects (they contain calls and environments) and
# compare everything else exactly.
comparable_nodes <- function(nodes) {
  lapply(nodes, function(nd) {
    if (!is.null(nd$info)) nd$info$object <- NULL
    nd
  })
}

# ---------------------------------------------------------------------------
# Extraction of the logp.supLM closure for the p-value grid

# Walk the body of mob_partynode to find the definition of logp.supLM inside
# mob_grow_fluctests, and evaluate it unchanged with the Hansen table bound.
extract_logp_suplm <- function() {
  find_assign <- function(expr, name) {
    if (is.call(expr)) {
      if (identical(expr[[1L]], as.name("<-")) && identical(expr[[2L]], as.name(name))) {
        return(expr[[3L]])
      }
      # Only recurse into calls: empty arguments such as in x[i, ] cannot be
      # passed on as function arguments.
      for (i in seq_along(expr)) {
        if (is.call(expr[[i]])) {
          hit <- find_assign(expr[[i]], name)
          if (!is.null(hit)) return(hit)
        }
      }
    }
    NULL
  }
  fluct <- find_assign(body(partykit:::mob_partynode), "mob_grow_fluctests")
  def <- find_assign(fluct, "logp.supLM")
  if (is.null(def)) stop("Could not find logp.supLM in partykit:::mob_partynode")
  env <- new.env(parent = asNamespace("partykit"))
  assign("beta", partykit:::mob_beta_suplm, envir = env)
  eval(def, env)
}
