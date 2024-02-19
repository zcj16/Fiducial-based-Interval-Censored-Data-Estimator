#include "utility.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace FICE {

    VectorXd block_i_element(const VectorXd& v, int i, int v_len) {
        int len_tmp = v_len - 1;
        VectorXd tmp(len_tmp);
        tmp.head(i) = v.head(i);
        tmp.tail(len_tmp - i) = v.tail(len_tmp - i);
        return tmp;
    }

    VectorXd linearInterpolation(VectorXd x, VectorXd y, VectorXd xcout) {
        vector<double> xtmp(x.data(), x.data() + x.size());
        VectorXd interpolated_y(xcout.size());
        vector<double> slope(x.size() - 1);
        for (int i = 0; i < x.size() - 1; ++i) {
            slope[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }
        for (int i = 0; i < xcout.size(); i++) {
            int low_index = 0;
            if (isinf(xcout[i]) || xcout[i] == INF) {
                interpolated_y[i] = 1;
            }
            else if (xcout[i] == 0) {
                interpolated_y[i] = 0;
            }
            else {
                while (x[low_index] < xcout[i] && low_index < x.size()) {
                    ++low_index;
                }
                /*cout << x[low_index] <<" " << xcout[i] << endl;*/
                //double y_inter = y[low_index - 1] + (xcout[i] - x[low_index - 1]) * (y[low_index] - y[low_index - 1]) / (x[low_index] - x[low_index - 1]);
                double y_inter = y[low_index - 1] + (xcout[i] - x[low_index - 1]) * slope[low_index - 1];

                interpolated_y[i] = y_inter;
            }
        }
        return interpolated_y;
    }

    vector<double> linearInterpolation(vector<double> x, vector<double> y, vector<double> xcout) {

        vector<double> interpolated_y(xcout.size());
        vector<double> slope(x.size() - 1);
        for (int i = 0; i < x.size() - 1; ++i) {
            slope[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }
        for (int i = 0; i < xcout.size(); i++) {
            int low_index = 0;
            if (isinf(xcout[i]) || xcout[i] == INF) {
                interpolated_y[i] = 1;
            }
            else if (xcout[i] == 0) {
                interpolated_y[i] = 0;
            }
            else {
                while (x[low_index] < xcout[i] && low_index < x.size()) {
                    ++low_index;
                }
                double y_inter = y[low_index - 1] + (xcout[i] - x[low_index - 1]) * slope[low_index - 1];

                interpolated_y[i] = y_inter;
            }
        }
        return interpolated_y;
    }


    vector<double> cond_distribution(VectorXd cdf, vector<double>& time_points, VectorXd l, VectorXd r) {
        int n_time = cdf.size();
        int n_sample = l.size();
        vector<double> pmf(n_time);
        vector<double> cond_pmf(n_time * n_sample, 0);
        pmf[0] = 0;
        for (int i = 1; i < n_time; ++i) {
            pmf[i] = cdf[i] - cdf[i - 1];
        }
        for (int i = 0; i < n_sample; ++i) {
            double sum_cond = 0;
            for (int j = 0; j < n_time; ++j) {// for every sample, cumulate the prob in its interval
                //cout << l[i] << " " << r[i] << " " << time_points[j] << endl;
                if (l[i] < time_points[j] && r[i] > time_points[j]) {
                    cond_pmf[i * n_time + j] = pmf[j];
                    sum_cond += pmf[j];
                }
            }

            if (sum_cond == 0) {
                int index = 0;
                for (int j = 0; j < n_time; ++j) {
                    if (r[i] > time_points[j]) {
                        index = max(index, j);
                    }
                }
                cond_pmf[index] = 1;
            }
            else {
                for (int j = 0; j < n_time; ++j) {
                    cond_pmf[i * n_time + j] = cond_pmf[i * n_time + j] / sum_cond;
                }
            }
        }
        return cond_pmf;
    }

    vector<double> cond_distribution(vector<double> cdf, vector<double>& time_points, vector<double> l, vector<double> r) {
        int n_time = cdf.size();
        int n_sample = l.size();
        vector<double> pmf(n_time);
        vector<double> cond_pmf(n_time * n_sample, 0);
        pmf[0] = 0;
        for (int i = 1; i < n_time; ++i) {
            pmf[i] = cdf[i] - cdf[i - 1];
        }
        for (int i = 0; i < n_sample; ++i) {
            double sum_cond = 0;
            for (int j = 0; j < n_time; ++j) {// for every sample, cumulate the prob in its interval
                //cout << l[i] << " " << r[i] << " " << time_points[j] << endl;
                if (l[i] < time_points[j] && r[i] > time_points[j]) {
                    cond_pmf[i * n_time + j] = pmf[j];
                    sum_cond += pmf[j];
                }
            }

            if (sum_cond == 0) {
                int index = 0;
                for (int j = 0; j < n_time; ++j) {
                    if (r[i] > time_points[j]) {
                        index = max(index, j);
                    }
                }
                cond_pmf[index] = 1;
            }
            else {
                for (int j = 0; j < n_time; ++j) {
                    cond_pmf[i * n_time + j] = cond_pmf[i * n_time + j] / sum_cond;
                }
            }
        }
        return cond_pmf;
    }

    void zeroInt(int* x, int length) {
        memset(x, 0, length * sizeof(int));
    }

    void zeroDouble(double* x, int length) {
        memset(x, 0, length * sizeof(double));
    }

    int imax2(int x, int y)
    {
        return (x < y) ? y : x;
    }

    double pack(const int nBits, const int* bits) {
        int i = nBits - 1;
        double pack = bits[i];
        for (i = nBits - 1; i > 0; --i) pack = 2.0 * pack + bits[i - 1];
        return(pack);
    }

    void unpack(const double pack, const int nBits, int* bits) {
        int i;
        double x = pack;
        for (i = 0; i <= nBits; ++i) {
            bits[i] = ((unsigned long)x & 1) ? 1 : 0;
            x = x / 2;
        }
    }

    void ibss(vector<double> LR, vector<double>& surv, vector<double> timepoints, vector<double>& timediff, vector<int>& inn,
        int nsample, int ntime, vector<double>& ibsN, int oob, double tau, int tree_idx) {
        int n, t;
        double err1, err2, SL, SR, b, timeSum1, timeSum2;
        for (n = 0; n < nsample * 2; ++n) ibsN[n] = 0.0;

        for (n = 0; n < nsample; ++n) {
            if (oob == 1)
                if (inn[n + tree_idx * nsample] > 0)
                    continue;

            err1 = 0.0;
            err2 = 0.0;
            timeSum1 = 0.0;
            timeSum2 = 0.0;

            // Getting SL and SR
            SL = 1.0;
            SR = 0.0; //default values
            for (t = 0; (t < ntime - 1) & (timepoints[t] <= tau); ++t) {
                if (LR[n] >= timepoints[t] && LR[n] < timepoints[t + 1]) {
                    SL = surv[n + t * nsample];
                }
                else if (LR[n + nsample] > timepoints[t] && LR[n + nsample] <= timepoints[t + 1]) {
                    SR = surv[n + t * nsample];
                }
            }
            if (SL <= SR) {
                b = 1.0;
            }
            else {
                b = 1.0 / (SL - SR);
            }

            for (t = 1; (t < ntime) & (timepoints[t] <= tau); ++t) {
                if (isfinite(timediff[t - 1]) == 0) break;
                if (timepoints[t] <= LR[n]) {
                    err1 += timediff[t - 1] * (1.0 - surv[n + t * nsample]) * (1.0 - surv[n + t * nsample]);
                    timeSum1 += timediff[t - 1];
                }
                else if (timepoints[t] > LR[n + nsample]) {
                    err1 += timediff[t - 1] * surv[n + t * nsample] * surv[n + t * nsample];
                    timeSum1 += timediff[t - 1];
                }
                else { // for ibs type 2
                    err2 += timediff[t - 1] * ((1.0 - b) * surv[n + t * nsample] + b * SR) * ((1.0 - b) * surv[n + t * nsample] + b * SR);
                    timeSum2 += timediff[t - 1];
                }
            }
            ibsN[n] = err1 / timeSum1;
            ibsN[n + nsample] = (err1 + err2) / (timeSum1 + timeSum2);
        }
    }

    double normCDF(double x) {
        return erf(x * 0.7071068) / 2.0 + 0.5;
    }

    // x: original time, y: original prob mass, n: length(x),
    // xp: grid timepoints, yp: a prob mass vector to be returned, np: length(xp)
    // bw: bandwidth
    // y and yp are in a (normalized) density scale and are not probability masses.
    // smoothing is done in a density scale and then the smoothed density is transformed back to prob scale.
    void ksmooth(vector<double>& x, vector<double>& y, int n,
        vector<double>& xp, vector<double>& yp, int np,
        double bw, int sumToOne, double cdfAtTau)
    {
        int imin = 0;
        double cutoff = 0.0, num, den, x0, w, lb, ub, lbInt;

        /* bandwidth is in units of half inter-quartile range. */
        //bw *= 0.3706506;
        cutoff = bw;
        int smooth = bw > 0.0001;

        while (x[imin] < xp[0] - cutoff && imin < n) imin++;

        yp[0] = 0.0;  // f(0) = 0.0 by default.
        for (int j = 1; j < np; j++) {
            num = den = 0.0;
            x0 = xp[j];
            lb = smooth ? x0 - cutoff : xp[j - 1];
            ub = smooth ? x0 + cutoff : x0;

            for (int i = imin; i < n; i++) {

                if (x[i] < lb) imin = i;
                else {

                    if (x[i] >= ub) {
                        lbInt = (x[i - 1] >= lb) ? x[i - 1] : lb;
                        w = smooth ? normCDF(cutoff / bw) - normCDF((lbInt - x0) / bw) : ub - lbInt;
                        num += y[i] * w;   // not just y * w but /(delta_x) to smooth in a density scale.
                        den += w;
                        break;
                    }
                    else {
                        if (x[i - 1] < lb) {
                            w = smooth ? normCDF((x[i] - x0) / bw) - normCDF(-cutoff / bw) : (x[i] - lb);
                        }
                        else {
                            w = smooth ? normCDF((x[i] - x0) / bw) - normCDF((x[i - 1] - x0) / bw) : (x[i] - x[i - 1]);
                        }
                        num += y[i] * w;  // not just y * w but /(delta_x) to smooth in a density scale.

                    }
                    //if (j==0) Rprintf("j=%d, imin = %d, (%2.3f,  %2.3f, %2.2f, %2.2f) = \n", j, imin, w, y[i], num, den);
                    //den = ub - lb;
                    den += w;
                }
            }
            if (den > 0) {
                yp[j] = num / den;  // transform the density scale back to prob mass scale
            }
            else if (num == 0) {  // This condition has been added! (tail of surv curve is just zero.)
                yp[j] = 0.0;
            }
            else {
                yp[j] = INF;
            }

        }

        //sumToOne
        if (sumToOne) {
            double ypCum = 0.0, yp2Cum = 0.0, ratio = 1.0; //non-smoothed vector with the same gridline.
            vector<double> yp2(np);
            ksmooth(x, y, n, xp, yp2, np, 0.0, 0, -1.0);
            yp2[0] = 0.0; //by default
            for (int i = 1; i < np - 1; i++) { //only up to (*np - 1)th element.
                ypCum += yp[i] * (xp[i] - xp[i - 1]);   //convert prob and cumsum.
                if (cdfAtTau == -1.0) yp2Cum += yp2[i] * (xp[i] - xp[i - 1]); //convert prob and cumsum.
            }
            if (cdfAtTau != -1.0) yp2Cum = cdfAtTau;
            ratio = ypCum > 1e-20 ? yp2Cum / ypCum : 1.0; // When density is all zero except the last point, normalization causes NaN.

            for (int i = 0; i < np - 1; i++) {
                yp[i] *= ratio;
            }
            yp[np - 1] = 1.0 - yp2Cum;  //yp[last] is the residual.
        }
    }

}
