# glm.fit fixtures. Each case simulates its own data, writes it as CSV and
# records glm.fit's output plus partykit's glmfit (and lmfit for gaussian)
# objective and scores on the same data.

glm_case <- function(name, description, seed, family, builder,
                     weights = FALSE, offset = FALSE, start = NULL,
                     epsilon = 1e-8, maxit = 25L, expect_converged = TRUE) {
  list(name = name, description = description, seed = seed, family = family,
    builder = builder, weights = weights, offset = offset, start = start,
    epsilon = epsilon, maxit = maxit, expect_converged = expect_converged)
}

all_glm_cases <- function() {
  list(
    glm_case("gaussian_identity", "Gaussian, identity link, two regressors.",
      20261011L, "gaussian(link = \"identity\")", function() {
        n <- 150L
        d <- data.frame(x0 = 1, x1 = rnorm(n), x2 = runif(n))
        d$y <- 1 + 2 * d$x1 - 1.5 * d$x2 + rnorm(n)
        d
      }),
    glm_case("gaussian_weighted", "Gaussian with non-integer prior weights.",
      20261012L, "gaussian(link = \"identity\")", function() {
        n <- 150L
        d <- data.frame(x0 = 1, x1 = rnorm(n), x2 = runif(n), w = runif(n, 0.5, 2))
        d$y <- 1 + 2 * d$x1 - 1.5 * d$x2 + rnorm(n) / sqrt(d$w)
        d
      }, weights = TRUE),
    glm_case("gaussian_rank_deficient", "Gaussian with x3 = x1 + x2, so one coefficient is aliased (NA).",
      20261013L, "gaussian(link = \"identity\")", function() {
        n <- 100L
        d <- data.frame(x0 = 1, x1 = rnorm(n), x2 = rnorm(n))
        d$x3 <- d$x1 + d$x2
        d$y <- 0.5 + d$x1 - d$x2 + rnorm(n)
        d
      }),
    glm_case("binomial_logit", "Binary response, logit link.",
      20261014L, "binomial(link = \"logit\")", function() {
        n <- 300L
        d <- data.frame(x0 = 1, x1 = rnorm(n), x2 = rnorm(n))
        d$y <- as.integer(runif(n) < plogis(-0.5 + 1.2 * d$x1 - 0.7 * d$x2))
        d
      }),
    glm_case("binomial_proportions_weighted", "Proportions with the number of trials as prior weights.",
      20261015L, "binomial(link = \"logit\")", function() {
        n <- 120L
        d <- data.frame(x0 = 1, x1 = rnorm(n), w = sample(1:20, n, replace = TRUE))
        d$y <- rbinom(n, d$w, plogis(0.3 - 0.8 * d$x1)) / d$w
        d
      }, weights = TRUE),
    glm_case("binomial_near_separation", "Near-separation: y = 1(x1 > 0) with the two points closest to 0 flipped, so the MLE is finite but the slope is large and fitted probabilities reach 0 and 1.",
      20261016L, "binomial(link = \"logit\")", function() {
        n <- 100L
        d <- data.frame(x0 = 1, x1 = rnorm(n))
        d$y <- as.integer(d$x1 > 0)
        d$y[which(d$x1 == min(d$x1[d$x1 > 0]))] <- 0L
        d$y[which(d$x1 == max(d$x1[d$x1 < 0]))] <- 1L
        d
      }),
    glm_case("binomial_not_converged", "Moderate separation with maxit = 3, so glm.fit stops without converging.",
      20261017L, "binomial(link = \"logit\")", function() {
        n <- 100L
        d <- data.frame(x0 = 1, x1 = rnorm(n))
        d$y <- as.integer(d$x1 + rnorm(n, sd = 0.15) > 0)
        d
      }, maxit = 3L, expect_converged = FALSE),
    glm_case("poisson_log", "Counts, log link, moderate mean.",
      20261018L, "poisson(link = \"log\")", function() {
        n <- 200L
        d <- data.frame(x0 = 1, x1 = rnorm(n), x2 = runif(n))
        d$y <- rpois(n, exp(0.5 + 0.4 * d$x1 + 0.6 * d$x2))
        d
      }),
    glm_case("poisson_zeros", "Counts with about two thirds zeros.",
      20261019L, "poisson(link = \"log\")", function() {
        n <- 200L
        d <- data.frame(x0 = 1, x1 = rnorm(n))
        d$y <- rpois(n, exp(-1.2 + 0.5 * d$x1))
        d
      }),
    glm_case("poisson_zeros_offset_weighted", "Counts with many zeros, an exposure offset and integer prior weights.",
      20261020L, "poisson(link = \"log\")", function() {
        n <- 200L
        d <- data.frame(x0 = 1, x1 = rnorm(n), exposure = runif(n, 0.2, 1.5),
          w = sample(1:3, n, replace = TRUE))
        d$offset <- log(d$exposure)
        d$y <- rpois(n, d$exposure * exp(-1 + 0.5 * d$x1))
        d
      }, weights = TRUE, offset = TRUE),
    glm_case("poisson_start", "Counts with many zeros fitted from a poor user-supplied start (2, -4). Same data as poisson_zeros.",
      20261019L, "poisson(link = \"log\")", function() {
        n <- 200L
        d <- data.frame(x0 = 1, x1 = rnorm(n))
        d$y <- rpois(n, exp(-1.2 + 0.5 * d$x1))
        d
      }, start = c(2, -4))
  )
}

glm_call_text <- function(case, xcols) {
  sprintf(
    "stats::glm.fit(x = as.matrix(d[, c(%s)]), y = d$y, weights = %s, start = %s, offset = %s, family = %s, control = stats::glm.control(epsilon = %s, maxit = %s))",
    paste(sprintf("\"%s\"", xcols), collapse = ", "),
    if (case$weights) "d$w" else "NULL",
    deparse_value(case$start),
    if (case$offset) "d$offset" else "NULL",
    case$family,
    deparse_value(case$epsilon),
    deparse_value(case$maxit)
  )
}

collect_warnings <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = unique(warnings))
}

glm_fixture <- function(case, data_dir) {
  set_fixture_seed(case$seed)
  d <- case$builder()
  xcols <- grep("^x[0-9]+$", names(d), value = TRUE)
  ds <- list(name = paste0("glm_", case$name), description = case$description,
    source = "simulated", seed = case$seed,
    code = paste(deparse(body(case$builder), width.cutoff = 100L), collapse = "\n"),
    data = d)
  write_dataset(ds, data_dir)

  call_text <- glm_call_text(case, xcols)
  env <- new.env(parent = globalenv())
  assign("d", d, envir = env)
  res <- collect_warnings(eval(parse(text = call_text)[[1L]], env))
  z <- res$value
  if (!identical(z$converged, case$expect_converged)) {
    stop(sprintf("glm case %s: converged = %s, expected %s", case$name, z$converged, case$expect_converged))
  }

  # partykit's fitter on the same data, with case weights as glmtree uses it.
  x <- as.matrix(d[, xcols])
  w <- if (case$weights) d$w else NULL
  off <- if (case$offset) d$offset else NULL
  fam <- eval(parse(text = case$family))
  pk <- collect_warnings(partykit:::glmfit(y = d$y, x = x, start = case$start,
    weights = w, offset = off, family = fam, epsilon = case$epsilon,
    maxit = case$maxit, estfun = TRUE, object = FALSE, caseweights = TRUE))$value
  lmf <- if (fam$family == "gaussian") {
    partykit:::lmfit(y = d$y, x = x, weights = w, offset = off, estfun = TRUE)
  } else NULL

  list(
    fixture = paste0("glm_", case$name),
    description = case$description,
    generator = "r/fixtures/glm.R",
    r_call = call_text,
    versions = fixture_versions(),
    data_csv = file.path("data", paste0("glm_", case$name, ".csv")),
    seed = case$seed,
    family = list(family = fam$family, link = fam$link),
    x_columns = arr(xcols),
    weights_column = if (case$weights) "w" else NULL,
    offset_column = if (case$offset) "offset" else NULL,
    start = if (is.null(case$start)) NULL else arr(case$start),
    control = list(epsilon = case$epsilon, maxit = case$maxit),
    warnings = arr(res$warnings),
    result = list(
      coefficient_names = arr(names(z$coefficients)),
      coefficients = arr(finite_or_na(z$coefficients)),
      deviance = z$deviance,
      null_deviance = z$null.deviance,
      aic = z$aic,
      iter = z$iter,
      converged = z$converged,
      boundary = z$boundary,
      rank = z$rank,
      df_residual = z$df.residual,
      df_null = z$df.null,
      fitted_values = arr(unname(z$fitted.values)),
      linear_predictors = arr(unname(z$linear.predictors)),
      working_weights = arr(unname(z$weights)),
      working_residuals = arr(unname(z$residuals)),
      prior_weights = arr(unname(z$prior.weights))
    ),
    partykit_glmfit = list(
      call = "partykit:::glmfit(y, x, start, weights, offset, family, epsilon, maxit, estfun = TRUE, object = FALSE, caseweights = TRUE)",
      objfun = pk$objfun,
      estfun = unname(pk$estfun)
    ),
    partykit_lmfit = if (is.null(lmf)) NULL else list(
      call = "partykit:::lmfit(y, x, weights, offset, estfun = TRUE)",
      coefficients = arr(finite_or_na(lmf$coefficients)),
      objfun = lmf$objfun,
      estfun = unname(lmf$estfun)
    )
  )
}
