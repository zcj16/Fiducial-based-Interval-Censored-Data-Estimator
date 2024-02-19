#pragma once
#pragma once
#ifndef FICE_FIDUCIAL_
#define FICE_FIDUCIAL_

#include "utility.h"
#include "eiquadprog.h"
//#include "eigen-QP.h"
//#include "EigenQP.h"
#include <float.h>
#include <utility>
#include "random.hpp"

//using namespace EigenQP;

namespace FICE {
	class Fiducial
	{
	public:
		//Constructer
		Fiducial(vector<double> left_observation,
			vector<double> right_observation,
			vector<double> test_grid = { log(4) - log(3),log(2),log(4) },
			const double grid_high = 3.0,
			const double alpha = 0.05,
			int seed = 1,
			const int n_mcmc = 1000,
			const int n_burn = 100,
			const int n_grid = 100,
			const double grid_low = 0.0,
			const double lambda = 1);

		// Estimator
		vector<vector<double>> Estimator();

	private:

		void GridInitializer();
		void GridBoundsInitializer();
		// Making n (same length with left_observation_) samples at Unif(0,1) and sorting according to (l + r)/2
		void GibbsInitializer();
		// Get the lower and upper bound of fiducial distribution
		pair<vector<double>, vector<double>> get_bounds();


		// Generate mu from the standard uniform distribution on a set described as r_i < l_j then mu_i < mu_j
		pair<vector<vector<double>>, vector<vector<double>>> GibbsSampler();

		// Fitting a continuous distribution function by using linear/quadratic interpolation via a quadratic programming
		vector<vector<double>> LinearInterpolation();

		vector<double> left_observation_; // Left bound of observation data
		vector<double> right_observation_; // Right bound of observation data
		double alpha_; // Alpha of Confidence interval
		int n_mcmc_; // Number of iterative times of Gibbs Sampler
		int n_burn_; // Number of burn-in of Gibbs Sampler
		int n_grid_; // Number of intervals of the Fiducial Grid
		double grid_low_; // Lower bound of grid interval
		double grid_high_; // Upper bound of grid interval
		int lambda_;
		int seed_; // Random number Generating Seed
		int n_sample_; // Samples' number is based on data's size
		vector<double> grid_; // Grid of Gibbs Sampler
		vector<double> test_grid_; //Test Grid of Gibbs Sampler
		vector<double> mu_values_; //Values of Gibbs Sampler
		vector<size_t> sorted_left_index_; // decreasing order of left observation
		vector<size_t> sorted_right_index_; // increasing order of right observation 
		map<size_t, vector<size_t>> grid_lower_bound_index;
		map<size_t, vector<size_t>> grid_upper_bound_index;
	};
}

#endif // !FICE_FIDUCIAL_
