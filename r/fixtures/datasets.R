# Datasets used by the tree fixtures. Each builder returns the data frame;
# make_dataset() records the seed and the builder's source in the schema file.

make_dataset <- function(name, description, source, seed, builder) {
  if (!is.null(seed)) set_fixture_seed(seed)
  list(
    name = name,
    description = description,
    source = source,
    seed = seed,
    code = paste(deparse(body(builder), width.cutoff = 100L), collapse = "\n"),
    data = builder()
  )
}

build_synth_diabetes <- function() {
  env <- new.env()
  utils::data("SynthDiabetes", package = "mlbench", envir = env)
  env$SynthDiabetes
}

# Gaussian responses with planted structure in x's coefficients.
# z1 numeric, z2 nominal, z3 ordered, z4 integer-valued numeric with many ties
# (noise), z5 nominal noise.
build_sim_gaussian <- function() {
  n <- 600L
  d <- data.frame(
    x = rnorm(n),
    z1 = runif(n, 0, 10),
    z2 = factor(sample(c("a", "b", "c", "d"), n, replace = TRUE)),
    z3 = factor(sample(paste0("L", 1:5), n, replace = TRUE),
      levels = paste0("L", 1:5), ordered = TRUE),
    z4 = as.numeric(sample(1:20, n, replace = TRUE)),
    z5 = factor(sample(c("u", "v", "w"), n, replace = TRUE))
  )
  e <- rnorm(n)
  # No instability.
  d$y_null <- 1 + 2 * d$x + e
  # One split: z1 <= 5.
  left <- d$z1 <= 5
  d$y_one <- ifelse(left, 1 + 2 * d$x, -1 + 0.5 * d$x) + e
  # Several splits: z1 <= 4, then z3 (ordered) on the left and z2 (nominal)
  # on the right.
  low3 <- as.integer(d$z3) <= 2L
  ac <- d$z2 %in% c("a", "c")
  b0 <- ifelse(d$z1 <= 4, ifelse(low3, 1, -1.5), ifelse(ac, 2, 2))
  b1 <- ifelse(d$z1 <= 4, ifelse(low3, 1, 1), ifelse(ac, -1, 1.5))
  d$y_multi <- b0 + b1 * d$x + e
  # Positive integer case weights, and a copy with about 10% zeros. partykit
  # 1.3-0 mishandles zero case weights in the instability tests (see
  # notes/phase1-questions.md), so w0 is used only in an edge-case fixture.
  d$w <- sample(1:3, n, replace = TRUE)
  d$w0 <- d$w * as.integer(runif(n) > 0.1)
  d
}

# Binary responses. z1 numeric, z2 nominal, z3 ordered, z4 numeric noise.
build_sim_binomial <- function() {
  n <- 800L
  d <- data.frame(
    x = rnorm(n),
    z1 = runif(n, -1, 1),
    z2 = factor(sample(c("p", "q", "r"), n, replace = TRUE)),
    z3 = factor(sample(c("lo", "mid", "high", "top"), n, replace = TRUE),
      levels = c("lo", "mid", "high", "top"), ordered = TRUE),
    z4 = rnorm(n)
  )
  u <- runif(n)
  d$y_null <- as.integer(u < plogis(-0.3 + 1 * d$x))
  d$y_one <- as.integer(u < plogis(ifelse(d$z1 <= 0, -1 + 1.5 * d$x, 1 - 1.5 * d$x)))
  eta <- ifelse(d$z2 == "r", -1 + 1.5 * d$x,
    ifelse(d$z1 <= 0.3, 0.8 - 1 * d$x, -0.8 + 0.5 * d$x))
  d$y_multi <- as.integer(u < plogis(eta))
  d
}

# Counts with an exposure offset. z1 numeric, z2 nominal, z3 ordered,
# z4 numeric noise.
build_sim_poisson <- function() {
  n <- 800L
  d <- data.frame(
    x = runif(n),
    z1 = runif(n, 0, 1),
    z2 = factor(sample(c("g1", "g2", "g3", "g4"), n, replace = TRUE)),
    z3 = factor(sample(c("s1", "s2", "s3", "s4", "s5"), n, replace = TRUE),
      levels = c("s1", "s2", "s3", "s4", "s5"), ordered = TRUE),
    z4 = rnorm(n),
    exposure = runif(n, 0.5, 2)
  )
  d$y_null <- rpois(n, d$exposure * exp(0.2 + 0.8 * d$x))
  high3 <- as.integer(d$z3) >= 4L
  eta <- ifelse(high3, 1.2 - 0.5 * d$x,
    ifelse(d$z1 <= 0.5, -0.5 + 1.5 * d$x, 0.3 + 0.3 * d$x))
  d$y_multi <- rpois(n, d$exposure * exp(eta))
  d$w <- sample(1:3, n, replace = TRUE)
  d
}

# The rows of a dataset with positive weight in `weight_col`, as a new dataset.
# removed_rows holds the 0-based indices (in the source data) of the dropped
# rows, so a fixture on the reduced data can be mapped back to the source.
drop_zero_weight_rows <- function(src, name, description, weight_col) {
  zero <- which(src$data[[weight_col]] == 0)
  stopifnot(length(zero) > 0L)
  list(
    name = name,
    description = description,
    source = "simulated",
    seed = src$seed,
    code = sprintf("subset of %s with rows where %s == 0 removed", src$name, weight_col),
    data = src$data[-zero, , drop = FALSE],
    source_dataset = src$name,
    removed_rows = zero - 1L
  )
}

all_datasets <- function() {
  sim_gaussian <- make_dataset("sim_gaussian",
    "Simulated gaussian data with no, one and several planted splits.",
    "simulated", 20261001L, build_sim_gaussian)
  list(
    make_dataset("synth_diabetes",
      "mlbench::SynthDiabetes, used by the partykit 1.3-0 mob vignette in place of PimaIndiansDiabetes.",
      "mlbench", NULL, build_synth_diabetes),
    sim_gaussian,
    drop_zero_weight_rows(sim_gaussian, "sim_gaussian_nonzero_w0",
      "sim_gaussian without the rows where w0 is zero. Same columns, row order otherwise unchanged.",
      "w0"),
    make_dataset("sim_binomial",
      "Simulated binary data with no, one and several planted splits.",
      "simulated", 20261002L, build_sim_binomial),
    make_dataset("sim_poisson",
      "Simulated count data with an exposure offset and planted splits.",
      "simulated", 20261003L, build_sim_poisson)
  )
}
