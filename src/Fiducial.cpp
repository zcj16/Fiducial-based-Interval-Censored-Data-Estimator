#include "Fiducial.h"
#include <iostream>

namespace FICE {
	Fiducial::Fiducial(
		vector<double> left_observation,
		vector<double> right_observation,
		vector<double> test_grid,
		const double grid_high,
		const double alpha,
		int seed,
		const int n_mcmc,
		const int n_burn,
		const int n_grid,
		const double grid_low,
		const double lambda)
		: left_observation_(left_observation)
		, right_observation_(right_observation)
		, seed_(seed)
		, alpha_(alpha)
		, n_mcmc_(n_mcmc)
		, n_burn_(n_burn)
		, grid_low_(grid_low)
		, grid_high_(grid_high)
		, lambda_(lambda)
		, test_grid_(test_grid)
	{
		n_grid_ = n_grid + 1;
		n_sample_ = left_observation_.size();
		GridInitializer();
		sorted_right_index_ = sort_indexes(right_observation_);
		sorted_left_index_ = sort_indexes_decrease(left_observation_);
		GridBoundsInitializer();
		GibbsInitializer();
	};

	void Fiducial::GridInitializer() {
		grid_.resize(n_grid_);
		double diff = grid_high_ / (n_grid_ - 1);
		for (size_t i = 0; i < n_grid_; i++) {
			grid_[i] = i * diff;
		}
	}

	void Fiducial::GridBoundsInitializer() {
		for (size_t i = 0; i < n_grid_; i++) {
			double step = grid_[i];
			for (size_t j = 0; j < n_sample_; j++) {
				if (step >= right_observation_[sorted_right_index_[j]]) {
					grid_lower_bound_index[i].push_back(sorted_right_index_[j]);
				}
				else {
					break;
				}
			}
			for (size_t j = 0; j < n_sample_; j++) {
				if (step < left_observation_[sorted_left_index_[j]]) {
					grid_upper_bound_index[i].push_back(sorted_left_index_[j]);
				}
				else {
					break;
				}
			}
		}
	}

	pair<vector<double>, vector<double>> Fiducial::get_bounds() {
		size_t n_grid = n_grid_;
		pair<vector<double>, vector<double>> bounds;
		bounds.first.resize(n_grid);
		bounds.second.resize(n_grid);
		for (int i = 0; i < n_grid; i++) {
			double lower_bound = 0;
			double upper_bound = 1;

			for (size_t low : grid_lower_bound_index[i])
				lower_bound = max(mu_values_[low], lower_bound);

			for (size_t up : grid_upper_bound_index[i])
				upper_bound = min(mu_values_[up], upper_bound);

			bounds.first[i] = lower_bound;
			bounds.second[i] = upper_bound;
		}
		return bounds;
	}

	void Fiducial::GibbsInitializer() {
		static mt19937 e(seed_);
		static uniform_real_distribution<double> unif(0, 1);

		vector<double> mid(n_sample_);
		vector<double> mu_tmp(n_sample_);
		for (size_t i = 0; i < n_sample_; i++) {
			mid[i] = left_observation_[i] + right_observation_[i];
			mu_tmp[i] = unif(e);
		}
		sort(mu_tmp.begin(), mu_tmp.end());
		vector<size_t> order = sort_indexes(mid);
		mu_values_.resize(n_sample_);
		for (size_t i = 0; i < n_sample_; i++) {
			mu_values_[order[i]] = mu_tmp[i];
		}
	}

	pair<vector<vector<double>>, vector<vector<double>>> Fiducial::GibbsSampler() {
		size_t n_grid = n_grid_;
		vector<vector<double>> fid_lower(n_mcmc_);
		vector<vector<double>> fid_upper(n_mcmc_);
		for (int i = 0; i < n_mcmc_; i++) {
			fid_lower[i].resize(n_grid);
			fid_lower[i].resize(n_grid);
		}

		double left_ith, right_ith;
		double unif_lower_bound, unif_upper_bound;
		std::map<size_t, vector<size_t>> lower_bound_index;
		std::map<size_t, vector<size_t>> upper_bound_index;
		for (size_t i = 0; i < n_sample_; i++) {
			right_ith = right_observation_[i];
			left_ith = left_observation_[i];

			for (size_t j = 0; j < n_sample_; j++) {
				if (sorted_left_index_[j] == i) {
					continue;
				}
				if (right_ith <= left_observation_[sorted_left_index_[j]]) {
					upper_bound_index[i].push_back(sorted_left_index_[j]);
				}
				else break;
			}
			for (size_t j = 0; j < n_sample_; j++) {
				if (sorted_right_index_[j] == i) {
					continue;
				}
				if (left_ith >= right_observation_[sorted_right_index_[j]]) {
					lower_bound_index[i].push_back(sorted_right_index_[j]);
				}
				else break;
			}
		}

		for (int j = 0; j < n_mcmc_ + n_burn_; j++) {
			std::random_device rd;
			std::default_random_engine e(rd());
			for (int i = 0; i < n_sample_; i++) {
				unif_lower_bound = 0;
				unif_upper_bound = 1;

				for (size_t low : lower_bound_index[i])
					unif_lower_bound = max(mu_values_[low], unif_lower_bound);

				for (size_t up : upper_bound_index[i])
					unif_upper_bound = min(mu_values_[up], unif_upper_bound);

				// Generate the ith number 
				nonstd::uniform_real_distribution<double> unif(unif_lower_bound, unif_upper_bound);
				mu_values_[i] = unif(e);
			}

			pair<vector<double>, vector<double>> tmp_bounds = get_bounds();
			if (j >= n_burn_) {
				fid_lower[j - n_burn_] = tmp_bounds.first;
				fid_upper[j - n_burn_] = tmp_bounds.second;
			}
		}
		return make_pair(fid_lower, fid_upper);
	}

	vector<vector<double>> Fiducial::LinearInterpolation() {
		pair<vector<vector<double>>, vector<vector<double>>> fid_grid = GibbsSampler();
		int n_grid = n_grid_;
		size_t n_test_gird = test_grid_.size();
		vector<vector<double>> inter_grid(n_mcmc_);
		for (size_t i = 0; i < n_mcmc_; i++) {
			inter_grid[i].resize(n_grid);
		}
		vector<vector<double>> interpolate_res(n_test_gird);
		for (size_t i = 0; i < n_test_gird; i++) {
			interpolate_res[i].resize(n_mcmc_);
		}

		double diff = 1 / pow((grid_high_ - grid_low_) * (n_grid) / (n_grid - 1), 2);

		MatrixXd quad_prod_mat = MatrixXd::Zero(n_grid, n_grid);
		quad_prod_mat(0, 0) = lambda_ + diff;
		quad_prod_mat(n_grid - 1, n_grid - 1) = lambda_ + diff;
		for (int i = 1; i < n_grid_ - 1; i++) {
			quad_prod_mat(i, i) = 2 * diff;
		}
		for (int i = 1; i < n_grid_; i++) {
			quad_prod_mat(i - 1, i) = -diff;
			quad_prod_mat(i, i - 1) = -diff;
		}

		MatrixXd Aeq = MatrixXd::Zero(n_grid, n_grid);
		VectorXd beq = VectorXd::Zero(n_grid);

		static mt19937_64 e(seed_);
		beta_distribution<double> beta(0.5, 0.5);

		MatrixXd A = MatrixXd::Zero(n_grid, 2 * n_grid);
		for (int i = 0; i < n_grid; i++) {
			A(i, i) = -1;
			A(i, i + n_grid) = 1;
		}


#pragma omp parallel for
		for (int i = 0; i < n_mcmc_; i++) {
			VectorXd fid_lower = Map<VectorXd, Unaligned>(fid_grid.first[i].data(), fid_grid.first[i].size());
			VectorXd fid_upper = Map<VectorXd, Unaligned>(fid_grid.second[i].data(), fid_grid.second[i].size());
			VectorXd quad_prod_lin(n_grid);
			double w0 = beta(e) * (fid_upper[0] - fid_lower[0]) + fid_lower[0];
			double wEnd = beta(e) * (fid_upper[n_grid - 1] - fid_lower[n_grid - 1]) + fid_lower[n_grid_ - 1];

			quad_prod_lin << lambda_ * -w0, lambda_* VectorXd::Zero(max(0, n_grid - 2)).array(), lambda_ * -wEnd;

			VectorXd x(n_grid);
			VectorXd b(2 * n_grid);
			b << fid_upper, fid_lower - 2 * fid_lower;
			//quadprog(quad_prod_mat, quad_prod_lin, A, b, x);
			solve_quadprog(quad_prod_mat, quad_prod_lin, Aeq, beq, A, b, x);
			//cout << x.transpose() << endl;
			//cout << endl;
			for (int j = 0; j < n_grid; j++) {
				if (x[j] < 0) {
					cout << "bad solution!" << endl;
				}
				inter_grid[i][j] = x[j];
			}
		}

		for (size_t i = 0; i < n_mcmc_; i++) {
			vector<double> interpolate_tmp = linearInterpolation(grid_, inter_grid[i], test_grid_);
			for (size_t j = 0; j < n_test_gird; j++) {
				interpolate_res[j][i] = interpolate_tmp[j];
			}
		}

		for (size_t i = 0; i < n_test_gird; i++) {
			sort(interpolate_res[i].begin(), interpolate_res[i].end());
		}

		return interpolate_res;
	}

	vector<vector<double>> Fiducial::Estimator() {
		double level = alpha_ / 2;
		size_t n_test_gird = test_grid_.size();
		vector<vector<double>> res(3);
		for (size_t i = 0; i < 3; i++) {
			res[i].resize(n_test_gird);
		}

		vector<vector<double>> interpolate_res = LinearInterpolation();
		//Point estimator
		for (size_t i = 0; i < n_test_gird; i++) {
			res[0][i] = interpolate_res[i][n_mcmc_ * 0.5];
		}
		// CI lower bound
		for (size_t i = 0; i < n_test_gird; i++) {
			res[1][i] = interpolate_res[i][n_mcmc_ * level];
		}
		// CI upper bound
		for (size_t i = 0; i < n_test_gird; i++) {
			res[2][i] = interpolate_res[i][n_mcmc_ * (1.0 - level)];
		}
		return res;
	}

	//MatrixXd FiducialEstimator::Estimator() {
	//	double level = alpha_ / 2;
	//	MatrixXd estimator(3, test_grid_.size());
	//	MatrixXd linefid = LinearInterpolation(grid_);
	//	// Point estimator
	//	estimator.row(0) = linefid.row(n_mcmc_ * 0.5);
	//	// CI lower bound
	//	//cout << estimator.row(1) << endl;
	//	estimator.row(1) = linefid.row(n_mcmc_ * level);
	//	//cout << estimator.row(2) << endl;
	//	// CI upper bound
	//	estimator.row(2) = linefid.row(n_mcmc_ * (1.0 - level));
	//	//cout << estimator.row(3) << endl;
	//	return estimator;
	//}
}