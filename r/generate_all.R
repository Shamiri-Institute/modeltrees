# Regenerate every fixture in tests/fixtures/ from R.
#
# Usage, from the repository root:
#   Rscript r/generate_all.R
#
# The output is deterministic: every random draw follows set_fixture_seed()
# and no timestamps or absolute paths are written.

all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", all_args, value = TRUE)
if (length(file_arg) != 1L) stop("Run this file with Rscript")
r_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
repo_dir <- dirname(r_dir)

# Activate the r/ renv project regardless of the working directory.
original_dir <- getwd()
setwd(r_dir)
source(file.path("renv", "activate.R"))
setwd(original_dir)

# Phase 4 node models: negbinfit, safelogitfit, multinomfit.
for (f in sort(list.files(file.path(r_dir, "extensions"), pattern = "[.]R$", full.names = TRUE))) {
  source(f)
}
for (f in c("common.R", "datasets.R", "trees.R", "glm.R", "pvalues.R", "extensions.R", "sandwich.R", "predict.R")) {
  source(file.path(r_dir, "fixtures", f))
}

out_dir <- file.path(repo_dir, "tests", "fixtures")
data_dir <- file.path(out_dir, "data")
written <- character()
emit <- function(x, rel_path) {
  write_fixture_json(x, file.path(out_dir, rel_path))
  written <<- c(written, rel_path)
  cat("wrote", rel_path, "\n")
}

# Datasets.
datasets <- c(all_datasets(), extension_datasets(), sandwich_datasets())
names(datasets) <- vapply(datasets, `[[`, "", "name")
for (ds in datasets) {
  write_dataset(ds, data_dir)
  written <- c(written, file.path("data", paste0(ds$name, c(".csv", ".schema.json"))))
}

# Trees.
instrumented <- build_instrumented_mob_partynode()
tree_specs <- all_tree_specs()
tree_index <- list()
for (spec in tree_specs) {
  fx <- tree_fixture(spec, datasets[[spec$dataset]], instrumented)
  rel <- file.path("trees", paste0(spec$name, ".json"))
  emit(fx, rel)
  tree_index[[length(tree_index) + 1L]] <- list(file = rel, fixture = fx)
}

# Extension trees (mob() with the r/extensions/ fit functions).
ext_tree_index <- list()
for (spec in all_extension_tree_specs()) {
  fx <- tree_fixture(spec, datasets[[spec$dataset]], instrumented)
  rel <- file.path("extension_trees", paste0(spec$name, ".json"))
  emit(fx, rel)
  ext_tree_index[[length(ext_tree_index) + 1L]] <- list(file = rel, fixture = fx)
  if (spec$name == "sim_separation_safelogit_plain") {
    check_safelogit_matches_glmtree(spec, datasets[[spec$dataset]], instrumented)
  }
  if (spec$name == "quine_negbin_intercept_deviance") {
    thesis <- thesis_quine_crosscheck(fx)
  }
}
# The multinomial tree with a leaf without the reference class.
noref_tree_index <- list()
for (spec in reference_absent_tree_specs()) {
  fx <- tree_fixture(spec, datasets[[spec$dataset]], instrumented)
  fx$reference_absent <- reference_absent_check(spec, fx, datasets[[spec$dataset]])
  rel <- file.path("extension_trees", paste0(spec$name, ".json"))
  emit(fx, rel)
  noref_tree_index[[length(noref_tree_index) + 1L]] <- list(file = rel, fixture = fx)
}
emit(c(list(
  fixture = "thesis_quine_crosscheck",
  description = paste("The thesis' negative binomial quine tree (trusch.md:1356-1368, summaries",
    "at 1394-1470), grown by party 1.0-1, against extension_trees/quine_negbin_intercept_deviance.json",
    "grown by partykit 1.3-0 with negbinfit(objective = \"deviance\"). party 1.0-1 itself",
    "does not build on this R version (see docs/extensions-spec.md), so the comparison is with",
    "the values printed in the thesis, at their printed precision."),
  generator = "r/fixtures/extensions.R",
  tree_fixture = "extension_trees/quine_negbin_intercept_deviance.json"), thesis),
  "thesis_quine_crosscheck.json")

# Clustered trees for the cluster-robust covariances, and sandwich::vcovCL on
# their leaves and on the leaves of the clustered trees above.
cluster_tree_index <- list()
for (spec in all_cluster_tree_specs()) {
  fx <- tree_fixture(spec, datasets[[spec$dataset]], instrumented)
  rel <- file.path("cluster_trees", paste0(spec$name, ".json"))
  emit(fx, rel)
  cluster_tree_index[[length(cluster_tree_index) + 1L]] <- list(file = rel, fixture = fx, spec = spec)
}
vcovcl_index <- list()
grown <- c(tree_index, cluster_tree_index)
names(grown) <- vapply(grown, function(e) e$fixture$fixture, "")
specs_by_name <- c(tree_specs, all_cluster_tree_specs())
names(specs_by_name) <- vapply(specs_by_name, `[[`, "", "name")
for (nm in names(VCOVCL_MODELS)) {
  spec <- specs_by_name[[nm]]
  fx <- vcovcl_fixture(spec, grown[[nm]]$fixture, datasets[[spec$dataset]])
  rel <- file.path("vcovcl", paste0(nm, ".json"))
  emit(fx, rel)
  vcovcl_index[[length(vcovcl_index) + 1L]] <- list(file = rel, fixture = fx)
}

# predict() on new data for three tree fixtures.
predict_index <- list()
for (nm in PREDICT_TREES) {
  spec <- specs_by_name[[nm]]
  fx <- predict_fixture(spec, grown[[nm]]$fixture, datasets[[spec$dataset]])
  rel <- file.path("predict", paste0(nm, ".json"))
  emit(fx, rel)
  predict_index[[length(predict_index) + 1L]] <- list(file = rel, fixture = fx)
}

# Node-level fits of the extension models.
node_fit_index <- list()
for (case in all_node_fit_cases()) {
  fx <- node_fit_fixture(case, datasets[[case$dataset]])
  rel <- file.path("node_fits", paste0(case$name, ".json"))
  emit(fx, rel)
  node_fit_index[[length(node_fit_index) + 1L]] <- list(file = rel, fixture = fx)
}

# glm.fit.
glm_index <- list()
for (case in all_glm_cases()) {
  fx <- glm_fixture(case, data_dir)
  written <- c(written, file.path("data", paste0(fx$fixture, c(".csv", ".schema.json"))))
  rel <- file.path("glm", paste0(fx$fixture, ".json"))
  emit(fx, rel)
  glm_index[[length(glm_index) + 1L]] <- list(file = rel, fixture = fx)
}

# p-values.
pv <- pvalue_fixture()
emit(pv, "pvalues.json")

# ---------------------------------------------------------------------------
# README

md_escape <- function(x) gsub("|", "\\|", x, fixed = TRUE)

readme <- c(
  "# Test fixtures",
  "",
  "Generated by `Rscript r/generate_all.R` from the repository root. Do not",
  "edit by hand: this README is also generated. R is needed only to",
  "regenerate; the Python tests read these files directly.",
  "",
  sprintf("Generated with R %s, partykit %s, strucchange %s, sandwich %s, mlbench %s, jsonlite %s, MASS %s, nnet %s.",
    fixture_versions()$R, fixture_versions()$partykit, fixture_versions()$strucchange,
    fixture_versions()$sandwich, fixture_versions()$mlbench, fixture_versions()$jsonlite,
    utils::packageDescription("MASS")$Version, utils::packageDescription("nnet")$Version),
  "",
  "## Conventions",
  "",
  "- Numbers are written with 17 significant digits in JSON and CSV, so every",
  "  double round-trips exactly under a correctly rounded parser.",
  "- `null` stands for R's `NA` or a non-finite value. In `split_search.objective`",
  "  it marks an inadmissible candidate (R: `Inf`). In test p-values it marks a",
  "  variable that was not tested (fewer than two distinct values in the node).",
  "- Row indices (`rows`) and variable indices (`variable_index`, `test.best`) are",
  "  0-based. Node ids, `kids`, `parent`, `fitted_node` and `partykit_varid` keep",
  "  partykit's 1-based numbering.",
  "- `estfun` is stored row-major: one inner array per row of the node, in",
  "  `rows` order. Columns are named in `estfun_names`.",
  "- In `control`, `\"Inf\"` is R's `Inf` and `null` is R's `NULL`.",
  "- `stop_reason` is one of `split`, `minsize` (fewer than 2 * minsize",
  "  observations), `maxdepth`, `test_failed` (the instability tests raised an",
  "  error, caught by partykit's `try()`), `not_significant` (adjusted p-value",
  "  above `alpha`), `no_admissible_split`, `pruned` (split found, then",
  "  removed by post-pruning), or `no_estfun` (the fit function returned no",
  "  scores, so partykit skipped the tests; only the extension models do this).",
  "- `test.status` is `ok`, `not_run` (node stopped before testing, partykit",
  "  stores no `test`), or `failed` (partykit stores `NA`).",
  "- Factor responses (binomial) are encoded by `glm.fit` as 0 for the first",
  "  level in `response_levels` and 1 otherwise.",
  "- A dataset derived by deleting rows has `source_dataset` and `removed_rows`",
  "  (0-based indices into the source data) in its schema, and the tree fixtures",
  "  grown on it repeat both fields. Row indices in such a fixture refer to the",
  "  reduced data.",
  "- A fixture grown with `cluster` has a top-level `cluster` field naming the",
  "  cluster column of its data CSV, and `n_clusters` in every node (the number of",
  "  distinct cluster ids among the node's rows). Cluster ids are in the CSV, so",
  "  a node's cluster vector is the cluster column at the node's `rows`.",
  "- Each dataset has a `.schema.json` with column types, factor levels in R's",
  "  order, the seed, the RNG kind and the generating code.",
  "- Trees grown by `partykit::mob()` with an extension fit function have",
  "  `function_name` `mob`, `fit_function`, `fit_args`, and per node a",
  "  `model_info` object with the fit's extra outputs (for example `theta` for",
  "  `negbinfit`, `separation` for `safelogitfit`, `classes_present` for",
  "  `multinomfit`). See docs/extensions-spec.md for their meaning.",
  "",
  "## Data",
  "",
  "| File | Rows | Seed | Description |",
  "| --- | --- | --- | --- |"
)
for (ds in datasets) {
  readme <- c(readme, sprintf("| `data/%s.csv` | %d | %s | %s |", ds$name, nrow(ds$data),
    if (is.null(ds$seed)) sprintf("none (%s)", ds$source) else ds$seed, md_escape(ds$description)))
}
readme <- c(readme, "",
  "The glm.fit fixtures each have their own `data/glm_<case>.csv`, listed with",
  "the fixture below.", "",
  "## Tree fixtures",
  "",
  "Generator: `r/fixtures/trees.R`. Each tree is grown twice, once by partykit",
  "unchanged and once with an instrumented copy of `partykit:::mob_partynode`",
  "that also records raw p-values, split-search objectives and the effective",
  "`minsize` (see `r/fixtures/common.R`). The two trees must be identical or",
  "generation stops.",
  "",
  "Each file holds the full call, every `mob_control` setting, and per node: id,",
  "parent, depth, kids, stop reason, rows, `nobs`, `minsize`, coefficients,",
  "objective, the instability tests (statistic, raw p-value, log raw p-value,",
  "adjusted p-value, trimming quantities), the split, every split candidate",
  "with its objective, and `estfun`. `fitted_node` gives the terminal node of",
  "each row. `decision_margins` reports how close the fixture is to an `alpha`",
  "or split-objective tie.",
  "")
for (entry in tree_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("### `%s`", entry$file), "",
    fx$description, "",
    sprintf("Data `%s`. %d inner and %d terminal nodes.%s", fx$data_csv,
      fx$summary$n_inner, fx$summary$n_terminal,
      if (is.null(fx$removed_rows)) "" else sprintf(" %d rows of `%s` removed.",
        length(fx$removed_rows), fx$source_dataset)), "",
    "```r", fx$r_call, "```", "")
}
readme <- c(readme,
  "## Extension tree fixtures",
  "",
  "Generator: `r/fixtures/extensions.R` (specs) and `r/fixtures/trees.R`",
  "(growing and extraction, as above). Same format as the tree fixtures, in",
  "their own directory `extension_trees/` so that tests of the core models",
  "can keep globbing `trees/`. The fit functions are in `r/extensions/`",
  "and are specified in docs/extensions-spec.md. The generator also checks",
  "that `sim_separation_safelogit_plain` is identical to the tree glmtree grows",
  "with `binomial()` on the same data and settings.",
  "")
for (entry in ext_tree_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("### `%s`", entry$file), "",
    fx$description, "",
    sprintf("Data `%s`. %d inner and %d terminal nodes.", fx$data_csv,
      fx$summary$n_inner, fx$summary$n_terminal), "",
    "```r", fx$r_call, "```", "")
}
readme <- c(readme,
  "## Reference-absent multinomial tree",
  "",
  "Generator: `r/fixtures/extensions.R`. Extension tree format, stored in",
  "`extension_trees/` with the extension trees above and listed here",
  "separately because one of its leaves lacks the reference class and has no",
  "scores (see docs/sandwich-fixtures.md, section 3). The extra field",
  "`reference_absent` has every root split-search candidate refitted side by",
  "side (classes present, status and objective of each side, which add up to",
  "the split-search objective) and the leaf without the reference class with",
  "its objective.",
  "")
for (entry in noref_tree_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("### `%s`", entry$file), "",
    fx$description, "",
    sprintf("Data `%s`. %d inner and %d terminal nodes.", fx$data_csv,
      fx$summary$n_inner, fx$summary$n_terminal), "",
    "```r", fx$r_call, "```", "")
}
readme <- c(readme,
  "## Clustered tree fixtures",
  "",
  "Generator: `r/fixtures/sandwich.R` (specs) and `r/fixtures/trees.R`. Same",
  "format as the tree fixtures, grown with `cluster = school` on",
  "`sim_cluster_models` for the cluster-robust covariance fixtures below. They",
  "are kept out of `trees/` and `extension_trees/` so that the suites globbing",
  "those directories keep their fixture counts.",
  "")
for (entry in cluster_tree_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("### `%s`", entry$file), "",
    fx$description, "",
    sprintf("Data `%s`. %d inner and %d terminal nodes.", fx$data_csv,
      fx$summary$n_inner, fx$summary$n_terminal), "",
    "```r", fx$r_call, "```", "")
}
readme <- c(readme,
  "## Cluster-robust covariance fixtures",
  "",
  "Generator: `r/fixtures/sandwich.R`. One file per clustered tree, with every",
  "leaf: `rows` (0-based), `n`, `k`, `n_clusters` (G), the tree's",
  "coefficients, the refit used for sandwich (`refit_call`, evaluated with",
  "`leaf` the data frame of the leaf's rows; its coefficients equal the tree's",
  "and the difference is recorded), `vcov_model` (`stats::vcov` of the refit),",
  "`bread` (`sandwich::bread`, n times the inverse Hessian), `hessian` (by",
  "hand, see `hessian_definition`) and `vcovcl`: `sandwich::vcovCL` with",
  "`type` HC0 and HC1 crossed with `cadjust` TRUE and FALSE, with the",
  "school clusters and with `cluster = NULL` (each row its own cluster). Every",
  "matrix is checked against the sandwich written by hand from the tree's",
  "`estfun` (docs/api-design.md 5.1); the generator stops on a relative",
  "difference above 1e-8. Matrices are row-major. All trees are unweighted.",
  "docs/sandwich-fixtures.md records the decisions.",
  "")
for (entry in vcovcl_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("- `%s`: %s Model `%s`, %d leaves, clusters per leaf %s.", entry$file,
      fx$description, fx$model, length(fx$leaves),
      paste(vapply(fx$leaves, function(l) as.character(l$n_clusters), ""), collapse = ", ")))
}
readme <- c(readme, "",
  "## predict fixtures",
  "",
  "Generator: `r/fixtures/predict.R`. New rows for a tree fixture, each built",
  "from the first training row of a node with some columns changed (`cases`:",
  "`case`, `target_node`, `source_row`, `changed`, `note`): numeric splits at",
  "the break, one step above it, at -Inf and Inf; ordered splits at the break",
  "level, the level above and absent levels; nominal levels absent from the",
  "node; the split variable missing; and per leaf the off-path partitioning",
  "variables missing, the regressor missing, the offset variable missing and",
  "the unchanged row. `newdata` holds the columns (factor labels; numbers with",
  "null for NA and the strings \"Inf\" and \"-Inf\"). `routing` is",
  "`deterministic` or `random` (partykit draws a kid at random for a missing",
  "value, a level with no kid, or a value outside `.bincode`'s intervals such",
  "as -Inf; `random_reason` says which), checked by predicting under 25 seeds.",
  "`expected` has R's `node`, `response` and `link` for the deterministic",
  "rows (null for random rows and for R's NA predictions). `failing_calls`",
  "and `unseen_level_calls` record calls R stops on, with the message.",
  "")
for (entry in predict_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("- `%s`: %s %d rows, %d routed at random.", entry$file, fx$description, fx$n,
      sum(vapply(fx$cases, function(c) c$routing == "random", TRUE))))
}
readme <- c(readme, "",
  "## Node-level fit fixtures",
  "",
  "Generator: `r/fixtures/extensions.R`. One fit of an extension model on a",
  "subset of a dataset, with everything needed to repeat it: `rows` (0-based)",
  "and `row_filter`, the design matrix `x` (row-major, columns `x_names`), `y`",
  "(labels for factor responses, with `response_levels`), `weights`, `offset`,",
  "`start` and `fit_args`. `result` holds the coefficients (null for NA),",
  "`objfun`, `estfun` (null when the fit returns none) and `model_info`.",
  "",
  "`checks` records the numerical verification of `estfun`, each enforced by",
  "the generator: agreement with the analytic score written by hand",
  "(`estfun_vs_analytic_max_rel`; exact for multinom, within the IRLS",
  "tolerance for the glm.fit-based models, whose working weights lag one",
  "iteration), with central finite differences of the per-observation",
  "log-likelihood (`fd_vs_estfun_max_rel`, `fd_vs_analytic_max_rel`), and",
  "column sums at the fit (`estfun_colsums`, relative to the column sums of",
  "absolute values). `objfun_vs_loglik_abs` checks the objective against the",
  "summed log-likelihood. For negbin, `theta_score_sum` is the theta score at",
  "the returned theta (not part of `estfun`).",
  "")
for (entry in node_fit_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("- `%s`: %s Data `%s`, %d rows.", entry$file, fx$description, fx$data_csv, fx$n))
}
readme <- c(readme, "",
  "## Thesis cross-check",
  "",
  "### `thesis_quine_crosscheck.json`",
  "",
  "The quine tree printed in the thesis (grown with party 1.0-1) against",
  "`extension_trees/quine_negbin_intercept_deviance.json`. Split variables, node sizes,",
  "intercepts, theta, test statistics and 1 - p agree at the printed",
  "precision; the generator stops otherwise. party 1.0-1 does not compile on",
  "R 4.6 (its C code uses the `Calloc` macro, which R removed), so the thesis",
  "printout is the reference.",
  "",
  "## glm.fit fixtures",
  "",
  "Generator: `r/fixtures/glm.R`. Each file records `glm.fit`'s coefficients,",
  "deviance, null deviance, AIC, iteration count, convergence and boundary",
  "flags, rank, fitted values, linear predictors, working weights and residuals,",
  "and warnings. It also records partykit's `glmfit` objective and scores",
  "(and `lmfit` for gaussian cases) on the same data.",
  "")
for (entry in glm_index) {
  fx <- entry$fixture
  readme <- c(readme,
    sprintf("### `%s`", entry$file), "",
    fx$description, "",
    sprintf("Data `%s`, seed %d. Converged: %s, iterations: %d.", fx$data_csv,
      fx$seed, fx$result$converged, fx$result$iter), "",
    "```r", fx$r_call, "```", "")
}
readme <- c(readme,
  "## p-value fixture",
  "",
  "### `pvalues.json`",
  "",
  "Generator: `r/fixtures/pvalues.R`. supLM log p-values from partykit's",
  "`logp.supLM`, taken unchanged from inside `partykit:::mob_partynode`, over",
  sprintf("%d combinations of trimming and k, each at %d statistics.",
    length(pv$suplm), length(pv$suplm[[1L]]$statistic)),
  "Also chi-squared log p-values for factor tests, the Bonferroni/Sidak",
  "adjustment on a few p-value vectors, and the Hansen table itself.",
  "",
  "```r", pv$r_call, "```", "")
writeLines(readme, file.path(out_dir, "README.md"), useBytes = TRUE)
written <- c(written, "README.md")

# Report files in tests/fixtures that this run did not produce.
present <- list.files(out_dir, recursive = TRUE)
stale <- setdiff(present, written)
if (length(stale)) {
  warning("Files in tests/fixtures not produced by this run: ", paste(stale, collapse = ", "))
}
cat("done:", length(written), "files\n")
