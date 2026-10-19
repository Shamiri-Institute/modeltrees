# p-value fixtures for the instability tests, independent of any tree.
#
# logp.supLM is a closure inside mob_partynode() (modelparty.R). It is pulled
# out unchanged by extract_logp_suplm() in common.R and evaluated over a grid
# of statistics, numbers of parameters and trimming values. The grid crosses
# every branch: tau = 0.5, tau <= 0.01, 0.49 <= tau < 0.5, the interpolation
# range, and k > 40 (Estrella approximation).

suplm_lambda <- function(nobs, trim, minsize) {
  # As in mob_grow_fluctests (modelparty.R:262-265).
  from <- if (trim > 1) trim else ceiling(nobs * trim)
  from <- max(from, minsize)
  to <- nobs - from
  list(from = from, to = to, lambda = ((nobs - from) * to) / (from * (nobs - to)))
}

pvalue_fixture <- function() {
  logp_suplm <- extract_logp_suplm()
  nobs <- 1000L
  minsize <- 1L
  trims <- c(0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1 / 3, 0.4,
    0.45, 0.48, 0.49, 0.495, 0.5)
  ks <- c(1:12, 15L, 20L, 25L, 30L, 39L, 40L, 41L, 45L, 60L)
  stats <- c(0.01, 0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40,
    50, 60, 80, 100, 150, 200, 300)

  cases <- list()
  for (trim in trims) {
    tl <- suplm_lambda(nobs, trim, minsize)
    tau <- if (tl$lambda < 1) tl$lambda else 1 / (1 + sqrt(tl$lambda))
    for (k in ks) {
      logp <- vapply(stats, function(s) logp_suplm(s, k, tl$lambda), 0)
      cases[[length(cases) + 1L]] <- list(
        trim = trim, nobs = nobs, minsize = minsize, from = tl$from, to = tl$to,
        lambda = tl$lambda, tau = tau, k = k,
        statistic = arr(stats),
        log_p_value = arr(finite_or_na(logp))
      )
    }
  }

  # Chi-squared p-values for factor variables (modelparty.R:346), with
  # df = k * (levels - 1).
  chisq <- list()
  for (k in c(1L, 2L, 3L, 5L)) {
    for (nlev in c(2L, 3L, 4L, 6L)) {
      df <- k * (nlev - 1L)
      chisq[[length(chisq) + 1L]] <- list(
        k = k, levels = nlev, df = df, statistic = arr(stats),
        log_p_value = arr(finite_or_na(stats::pchisq(stats, df, lower.tail = FALSE, log.p = TRUE)))
      )
    }
  }

  # The multiplicity adjustment in mob_grow (modelparty.R:553-557).
  adjust <- function(p) {
    m <- sum(!is.na(p))
    ifelse(!is.na(p) & p > 0.001, 1 - (1 - p)^m, pmin(1, m * p))
  }
  raw_sets <- list(
    c(0.5, 0.04, 0.0009, 0.2, 0.9),
    c(1e-10, 0.001, 0.0010001, 0.3),
    c(0.0005, NA, 0.7),
    c(0.02)
  )
  bonferroni <- lapply(raw_sets, function(p) list(
    p_value_raw = arr(finite_or_na(p)),
    p_value_adjusted = arr(finite_or_na(adjust(p)))
  ))

  list(
    fixture = "pvalues",
    description = "supLM log p-values from partykit's logp.supLM, chi-squared log p-values, and the Bonferroni/Sidak adjustment.",
    generator = "r/fixtures/pvalues.R",
    r_call = "logp.supLM(statistic, k, lambda) extracted from partykit:::mob_partynode; lambda from (nobs, trim, minsize) as in mob_grow_fluctests",
    versions = fixture_versions(),
    hansen_table = list(
      description = "partykit:::mob_beta_suplm: 1000 x 3, rows (k - 1) * 25 + 1 .. k * 25 hold the 25 tau rows for k = 1..40; columns are two polynomial coefficients and the chi-squared df.",
      values = unname(as.matrix(partykit:::mob_beta_suplm))
    ),
    suplm = cases,
    chisq = chisq,
    bonferroni = bonferroni
  )
}
