#ifdef _OPENMP
#include <omp.h>
#endif

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <stdlib.h>
#include "utility.h"
#include "Fiducial.h"
using namespace Rcpp;
using namespace FICE;

// [[Rcpp::export]]
Rcpp::List Fiducial_Estimator(const std::vector<double> & l, 
                                   const std::vector<double> & r, 
                                   std::vector<double> &timepoints,
                                   double tau, double alpha){
  Fiducial F(l, r, timepoints, tau, alpha);
  vector<vector<double>> estimator = F.Estimator();
  List res = List::create(Named("point_estimator") = estimator[0] , 
                          _["CI_upper"] = estimator[1],
                         _["CI_lower"] = estimator[2]);
    
  return res;
}

