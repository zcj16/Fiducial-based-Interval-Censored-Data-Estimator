#pragma once
#pragma once
#ifndef  FICE_UTILITY_H_
#define  FICE_UTILITY_H_

#include <stdlib.h>
#include <vector>
#include <random>
#include <cstddef>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include <memory>
#include <limits>

#include "beta_distribution.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>

/* test if the bit at position pos is turned on */
#define isBitOn(x,pos) (((x) & (1 << (pos))) > 0)
/* swap two integers */
#define swapInt(a, b) ((a ^= b), (b ^= a), (a ^= b))

/* maximum number of categories allowed in categorical predictors */
#define MAX_CAT 53
/* Node status */
#define NODE_TERMINAL -1
#define NODE_TOSPLIT  -2
#define NODE_INTERIOR -3
#define INF std::numeric_limits<double>::infinity()

using namespace Eigen;
using namespace std;


namespace FICE {

	// Normal function, putting declaration in .h and definition in .cpp

	//  Get the vector without ith element
	VectorXd block_i_element(const VectorXd& v, int i, int v_len);
	VectorXd linearInterpolation(VectorXd x, VectorXd y, VectorXd xcout);
	vector<double> linearInterpolation(vector<double> x, vector<double> y, vector<double> xcout);
	// Initialize vector with 0
	void zeroInt(int* x, int length);
	void zeroDouble(double* x, int length);
	vector<double> cond_distribution(VectorXd cdf, vector<double>& time_points, VectorXd  l, VectorXd r);
	vector<double> cond_distribution(vector<double> cdf, vector<double>& time_points, vector<double> l, vector<double> r);

	void ksmooth(vector<double>& x, vector<double>& y, int n,
		vector<double>& xp, vector<double>& yp, int np,
		double bw, int sumToOne, double cdfAtTau);

	double pack(const int nBits, const int* bits);
	void unpack(const double pack, const int nBits, int* bits);

	int imax2(int x, int y);


	/* Permute the OOB part of a variable in x.
	 * Argument:
	 *   m: the variable to be permuted
	 *   x: the data matrix (variables in rows)
	 *   in: vector indicating which case is OOB
	 *   nsample: number of cases in the data
	 *   mdim: number of variables in the data
	 */
	void permuteOOB(int m, vector<double>& x, vector<int>& in, int nsample, int mdim);

	// measure functions
	void ibss(vector<double> LR, vector<double>& surv, vector<double> timepoints, vector<double>& timediff, vector<int>& inn,
		int nsample, int ntime, vector<double>& ibsN, int oob, double tau, int tree_idx);
	void testErr(vector<double> trueSurv, double* surv, vector<double> timepoints, vector<double>& timediff,
		int ntest, int ntime, double* interrts, double tau);

	// Template functions, putting in head file since it cannot compile when it depart from definition

	template<typename Func>
	struct condition_as_visitor_wrapper : Func {
		condition_as_visitor_wrapper(const Func& f) : Func(f) {}
		template<typename S, typename I>
		void init(const S& v, I i, I j) { return Func::operator()(v, i, j); }
	};

	// Catching the index in a Matrix, which fullfill some condition
	template<typename Mat, typename Func>
	void visit_condition(const Mat& m, const Func& f)
	{
		condition_as_visitor_wrapper<Func> visitor(f);
		m.visit(visitor);
	}

	// Getting the index of sorting
	template <typename T>
	vector<size_t> sort_indexes(const MatrixBase<T>& v) {
		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

		return idx;
	}

	// Getting the index of sorting
	template <typename T>
	vector<size_t> sort_indexes(const vector<T>& v) {
		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

		return idx;
	}
	// Getting the index of sorting
	template <typename T>
	vector<size_t> sort_indexes_decrease(const MatrixBase<T>& v) {
		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

		return idx;
	}

	// Getting the index of sorting
	template <typename T>
	vector<size_t> sort_indexes_decrease(const vector<T>& v) {
		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

		return idx;
	}

	template<typename Iter, typename RandomGenerator>
	Iter select_randomly(Iter start, Iter end, RandomGenerator* g) {
		std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
		std::advance(start, dis(*g));
		return start;
	}

	template<typename Iter>
	Iter select_randomly(Iter start, Iter end) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		return select_randomly(start, end, &gen);
	}

}

#endif // !FICE_UTILITY_H_
