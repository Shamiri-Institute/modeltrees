args <- commandArgs(trailingOnly = TRUE)
all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", all_args, value = TRUE)

if (length(file_arg) != 1L) {
  stop("Could not determine the path to smoke.R")
}

script_path <- normalizePath(sub("^--file=", "", file_arg))
project_dir <- dirname(script_path)
original_dir <- getwd()

# R loads startup files before it knows which script will run. Temporarily move
# into r/ so its renv activation script selects the correct project library.
setwd(project_dir)
source(file.path("renv", "activate.R"))
setwd(original_dir)

set.seed(20261019)

n <- 400L
dat <- data.frame(
  x = rnorm(n),
  z = runif(n, -1, 1)
)
linear_predictor <- ifelse(
  dat$z <= 0,
  -1.2 + 0.9 * dat$x,
  1.2 - 0.9 * dat$x
)
dat$y <- rbinom(n, size = 1L, prob = plogis(linear_predictor))

tree <- partykit::glmtree(
  y ~ x | z,
  data = dat,
  family = binomial(),
  minsize = 60L
)

data_names <- names(partykit::data_party(tree))
terminal_ids <- partykit::nodeids(tree, terminal = TRUE)
internal_ids <- setdiff(partykit::nodeids(tree), terminal_ids)
splits <- partykit::nodeapply(
  tree,
  ids = internal_ids,
  FUN = function(node) {
    split <- partykit::split_node(node)
    list(
      node = partykit::id_node(node),
      variable = data_names[[partykit::varid_split(split)]],
      split_points = unname(split$breaks)
    )
  }
)

if (length(splits) == 0L) {
  stop("The simulated data unexpectedly produced no split")
}

node_coefficients <- lapply(terminal_ids, function(id) {
  values <- stats::coef(tree, node = id)
  list(
    node = unname(id),
    coefficients = as.list(
      stats::setNames(as.numeric(values), names(values))
    )
  )
})

result <- list(
  family = "binomial",
  formula = "y ~ x | z",
  splits = unname(splits),
  node_coefficients = node_coefficients
)

output_path <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path(project_dir, "smoke.json")
}

jsonlite::write_json(
  result,
  path = output_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = NA,
  null = "null"
)

cat("Wrote", normalizePath(output_path), "\n")
