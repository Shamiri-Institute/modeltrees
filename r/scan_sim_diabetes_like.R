# Seed scan for the sim_diabetes_like dataset (r/fixtures/datasets.R).
#
# Usage, from the repository root:
#   Rscript r/scan_sim_diabetes_like.R [first_seed] [n_seeds]
#
# For each seed, grows the four sim_diabetes_like_* trees of r/fixtures/trees.R
# and prints the seeds where
#   - the default tree splits on age and pedigree only, with 3 leaves,
#   - the BIC and AIC pruned trees differ,
#   - the AIC pruned trees with and without dfsplit differ.
# The first such seed is recorded as SIM_DIABETES_LIKE_SEED.

all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", all_args, value = TRUE)
if (length(file_arg) != 1L) stop("Run this file with Rscript")
r_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))

original_dir <- getwd()
setwd(r_dir)
source(file.path("renv", "activate.R"))
setwd(original_dir)

source(file.path(r_dir, "fixtures", "common.R"))
source(file.path(r_dir, "fixtures", "datasets.R"))

args <- commandArgs(trailingOnly = TRUE)
first <- if (length(args) >= 1L) as.integer(args[[1L]]) else 20261051L
count <- if (length(args) >= 2L) as.integer(args[[2L]]) else 50L

f <- diabetes ~ glucose + mass | pregnant + pressure + triceps + insulin + pedigree + age

split_vars <- function(tree) {
  ids <- partykit::nodeids(tree)
  inner <- setdiff(ids, partykit::nodeids(tree, terminal = TRUE))
  vapply(inner, function(i) names(tree$data)[partykit::split_node(tree[[i]]$node)$varid], "")
}
n_leaves <- function(tree) length(partykit::nodeids(tree, terminal = TRUE))

found <- integer()
for (seed in seq.int(first, length.out = count)) {
  set_fixture_seed(seed)
  d <- build_sim_diabetes_like()
  grow <- function(...) partykit::glmtree(f, data = d, family = binomial(), ...)
  default <- grow()
  big <- list(minsize = 50L, maxdepth = 4L, alpha = 0.9)
  bic <- do.call(grow, c(big, list(prune = "BIC")))
  aic <- do.call(grow, c(big, list(prune = "AIC")))
  aic_nodf <- do.call(grow, c(big, list(prune = "AIC", dfsplit = FALSE)))
  sv <- split_vars(default)
  leaves <- c(default = n_leaves(default), bic = n_leaves(bic), aic = n_leaves(aic),
    aic_nodfsplit = n_leaves(aic_nodf))
  ok <- n_leaves(default) == 3L && setequal(sv, c("age", "pedigree")) &&
    leaves[["bic"]] >= 3L && leaves[["bic"]] != leaves[["aic"]] &&
    leaves[["aic"]] != leaves[["aic_nodfsplit"]]
  cat(sprintf("seed %d: default splits %s; leaves %s%s\n", seed, paste(sv, collapse = ", "),
    paste(names(leaves), leaves, sep = " = ", collapse = ", "), if (ok) "  <- ok" else ""))
  if (ok) found <- c(found, seed)
}
cat("seeds that qualify:", if (length(found)) paste(found, collapse = ", ") else "none", "\n")
