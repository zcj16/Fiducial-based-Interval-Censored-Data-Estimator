#' Fiducial Estimaator
#' 
#' Estimate the interval censored data [L,R] to estimate the cdf function P[Y <= t]
#' 
#' @param L,R The interval censored data. L < R. It shoule be two numeric vectors.
#' @param timepoints The interesting timepoints you want to estimate. Default is all timepoints
#' from L,R
#' @param timeupper The upper bound of the test timepoints. Default is the max finite time of R.
#' @param alpha Significant level
#' @return The estimated cumulative density
#' 
#' @export
Fiducial <- function(L,R,timepoints = NULL, timeupper = NULL, alpha = 0.95){
  if (!is.null(L)|| !is.null(R)){
    if(length(L) != length(R)) stop("L and R should have same length")
    if(any(L > R)) stop("R should be greater or equal to L")
    if (any(L < 0) || any(R < 0)) stop("All values of L and R should be no less than 0")
  }else{
    stop("L and R should not be NULL.")
  }
  if(is.null(timeupper)){
    timeupper = max(R[is.finite(R)])
  }
  if(is.null(timepoints)){
    tmp_time <- sort(unique(0,L,R))
    timepoints <- c(tmp_time[tmp_time <= timeupper], Inf)
  }
  a <- 1 - alpha
  res <- Fiducial_Estimator(as.double(L),as.double(R),as.double(timepoints),as.double(timeupper), as.double(a))
  class(res) <- "FICE"
  return(res)
}