# Time partykit's lmtree and glmtree on the data of scripts/bench_tree.py.
#
# Usage, from the repository root:
#
#   uv run python scripts/bench_tree.py --full --write-csv DIR
#   Rscript scripts/bench_r.R DIR [runs] [sizes...]
#
# DIR holds bench_<n>.csv. runs defaults to 3, sizes to 1000 and 5000 (pass
# 20000 explicitly for the full run). Each cell gets one untimed warm-up run.
# The settings match the Python benchmark: y ~ x | z1 + ... + z6, partykit's
# defaults (alpha 0.05, Bonferroni, minsize 20, no pruning), binomial glmtree.

all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", all_args, value = TRUE)
if (length(file_arg) != 1L) {
  stop("Could not determine the path to bench_r.R")
}
script_path <- normalizePath(sub("^--file=", "", file_arg))
project_dir <- file.path(dirname(dirname(script_path)), "r")
original_dir <- getwd()

# As in r/smoke.R: move into r/ so renv activates the project library.
setwd(project_dir)
source(file.path("renv", "activate.R"))
setwd(original_dir)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("usage: Rscript scripts/bench_r.R DIR [runs] [sizes...]")
}
data_dir <- args[[1L]]
runs <- if (length(args) >= 2L) as.integer(args[[2L]]) else 3L
sizes <- if (length(args) >= 3L) as.integer(args[-(1:2)]) else c(1000L, 5000L)

read_bench <- function(n) {
  d <- utils::read.csv(file.path(data_dir, sprintf("bench_%d.csv", n)))
  d$z5 <- factor(d$z5, levels = 1:5, labels = paste0("o", 1:5), ordered = TRUE)
  d$z6 <- factor(d$z6, levels = 1:5, labels = letters[1:5])
  d
}

# Depth with the root at 1, as in the Python benchmark.
node_depth <- function(node) {
  kids <- partykit::kids_node(node)
  if (is.null(kids)) 1L else 1L + max(vapply(kids, node_depth, integer(1)))
}

fits <- list(
  lm = function(d) {
    partykit::lmtree(y_gauss ~ x | z1 + z2 + z3 + z4 + z5 + z6, data = d)
  },
  glm = function(d) {
    partykit::glmtree(
      y_bin ~ x | z1 + z2 + z3 + z4 + z5 + z6,
      data = d,
      family = binomial()
    )
  }
)

cat(sprintf(
  "R %s, partykit %s, runs per cell %d (after one warm-up)\n\n",
  getRversion(), utils::packageVersion("partykit"), runs
))
cat("model  n       nodes leaves depth  min s   median s  max s\n")
for (n in sizes) {
  d <- read_bench(n)
  for (name in names(fits)) {
    tree <- suppressWarnings(fits[[name]](d))
    times <- vapply(seq_len(runs), function(i) {
      t0 <- proc.time()[["elapsed"]]
      suppressWarnings(fits[[name]](d))
      proc.time()[["elapsed"]] - t0
    }, numeric(1))
    cat(sprintf(
      "%-6s %-7d %-5d %-6d %-5d %-7.3f %-9.3f %.3f\n",
      name, n, length(tree), partykit::width(tree),
      node_depth(partykit::node_party(tree)),
      min(times), stats::median(times), max(times)
    ))
  }
}
