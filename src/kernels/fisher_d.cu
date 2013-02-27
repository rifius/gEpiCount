/*
 * fisher_p.c
 *
 *  Created on: 22/06/2010
 *      Author: carlos
 *
 * Compute p-value for Fisher's exact test on a 2 x 2 contingency table.
 *
 * Code adapted from R and from Christine Loader
 * (see R source code and C.L. fast binomial distribution paper)
 * Functions have been reduced to their minimal expression, integer arguments
 * used whenever possible considering it is being used in an integer 2x2
 * table.
 *
 * Returns only logs of pvals.
 ********
 *  Copyright (c) 2012 Carlos Riveros
 *
 *  This file is part of gEpiCount.
 *
 *  gEpiCount is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gEpiCount is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gEpiCount.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef		_FISHER_D_CU_
#define		_FISHER_D_CU_

#include <math.h>
#include <math_constants.h>

#include	"../proc_gpu.h"

#ifndef M_2PI
#define M_2PI           6.283185307179586476925286766559        /* 2*pi */
#endif
#define	LOG2PI			1.8378770664093453390819377091247588396072387695312

//#define		ALTERNATE_DHYPER

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 130)
  #define		MY_NAN			CUDART_NAN_F
  #define		MY_INF			CUDART_INF_F
  #define		DOUBLE_			double
#else
  #define		MY_NAN			CUDART_NAN
  #define		MY_INF			CUDART_INF
  #define		DOUBLE_			double
#endif


/*
 *  DESCRIPTION
 *
 *    Computes the log of the error term in Stirling's formula.
 *      For n > 15, uses the series 1/12n - 1/360n^3 + ...
 *      For n <=15, integers or half-integers, uses stored values.
 *      For other n < 15, uses lgamma directly (don't use this to
 *        write lgamma!)
 *
 * Merge in to R:
 * Copyright (C) 2000, The R Core Development Team
 * R has lgammafn, and lgamma is not part of ISO C
 *
 * In any case, gamma functions are not used for whole number arguments.
 * The function has been reduced to the cases where the args are integer.
 */

/* stirlerr(n) = log(n!) - log( sqrt(2*pi*n)*(n/e)^n )
 *             = log Gamma(n+1) - 1/2 * [log(2*pi) + log(n)] - n*[log(n) - 1]
 *             = log Gamma(n+1) - (n + 1/2) * log(n) + n - log(2*pi)/2
 *
 * see also lgammacor() in ./lgammacor.c  which computes almost the same!
 */
//  Predefined errors
__device__ __constant__ DOUBLE_ sferr[16] = { 0.0, /* n=0 - wrong, place holder only */
		0.0810614667953272582196702, /* 1.0 */
		0.0413406959554092940938221, /* 2.0 */
		0.02767792568499833914878929, /* 3.0 */
		0.02079067210376509311152277, /* 4.0 */
		0.01664469118982119216319487, /* 5.0 */
		0.01387612882307074799874573, /* 6.0 */
		0.01189670994589177009505572, /* 7.0 */
		0.010411265261972096497478567, /* 8.0 */
		0.009255462182712732917728637, /* 9.0 */
		0.008330563433362871256469318, /* 10.0 */
		0.007573675487951840794972024, /* 11.0 */
		0.006942840107209529865664152, /* 12.0 */
		0.006408994188004207068439631, /* 13.0 */
		0.005951370112758847735624416, /* 14.0 */
		0.005554733551962801371038690 /* 15.0 */
};

__device__ DOUBLE_ stirlerr(int n) {

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

	if (n <= 15) {
		return (sferr[n]);
	}

	DOUBLE_ nn = (DOUBLE_) n * (DOUBLE_) n;
	if (n > 500)
		return ((S0 - S1 / nn) / (DOUBLE_) n);
	if (n > 80)
		return ((S0 - (S1 - S2 / nn) / nn) / (DOUBLE_) n);
	if (n > 35)
		return ((S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / (DOUBLE_) n);
	/* 15 < n <= 35 : */
	return ((S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / (DOUBLE_) n);
}

#undef S0
#undef S1
#undef S2
#undef S3
#undef S4

/*
 * bd0
 *
 * From R's bd0.c
 */
__device__ DOUBLE_ bd0(DOUBLE_ x, DOUBLE_ np) {
	DOUBLE_ ej, s, s1, v;
	int j;

	if (fabs(x - np) < 0.1 * (x + np)) {
		v = (x - np) / (x + np);
		s = (x - np) * v;/* s using v -- change by MM */
		ej = 2 * x * v;
		v = v * v;
		for (j = 1;; j++) { /* Taylor series */
			ej *= v;
			s1 = s + ej / ((j << 1) + 1);
			if (s1 == s) /* last term was effectively 0 */
				return (s1);
			s = s1;
		}
	}
	/* else:  | x - np |  is not too small */
	return (x * log(x / np) + np - x);
}

/*
 * dbinom_raw
 *
 * From R's dbinom.c.  Returns only log of values.
 */
__device__ DOUBLE_ dbinom_raw(int x, int n, DOUBLE_ p, DOUBLE_ q) {
	DOUBLE_ lf, lc;

	if (x == 0) {
		if (n == 0)
			return 0.0; // : 1.0;
		lc = (p < 0.1) ? -bd0((DOUBLE_) n, (DOUBLE_) n * q) - (DOUBLE_) n * p : (DOUBLE_) n * log(q);
		return lc; // : exp(lc);
	}
	if (x == n) {
		lc = (q < 0.1) ? -bd0((DOUBLE_) n, (DOUBLE_) n * p) - (DOUBLE_) n * q : (DOUBLE_) n * log(p);
		return lc; // : exp(lc);
	}
	if (x < 0 || x > n)
		return -MY_INF; // : 0.0;

	/* n*p or n*q can underflow to zero if n and p or q are small.  This
	 used to occur in dbeta, and gives NaN as from R 2.3.0.  */
	lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0((DOUBLE_) x, (DOUBLE_) n * p) - bd0((DOUBLE_) (n - x), (DOUBLE_) n * q);

	/* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
	/* Upto R 2.7.1:
	 * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
	 * -- following is much better for  x << n :
	 **/
//	lf = log(M_2PI) + log((DOUBLE_) x) + log1p(-(DOUBLE_) x / (DOUBLE_) n);
	lf = LOG2PI + log((DOUBLE_) x) + log1p(-(DOUBLE_) x / (DOUBLE_) n);

	lc -= 0.5 * lf;
	return (lc); // : exp(lc);
}

/*
 * dhyper
 *
 * From R's dhyper.c.  It is returning logarithm of the density distribution
 */
__device__ DOUBLE_ dhyper(int x, int r, int b, int n) {
	DOUBLE_ p, q;
	if (n < x || r < x || n - x > b) {
		return -MY_INF; // 0.0;
	}
	if (n == 0) {
		if (x == 0)
			p = 0.0; // 1.0;
		else
			p = -MY_INF; // 0.0;
		return p;
	}
	p = (DOUBLE_) n / (DOUBLE_) (r + b);
	q = (DOUBLE_) (r + b - n) / (DOUBLE_) (r + b);

	DOUBLE_ p1 = dbinom_raw(x, r, p, q);
	DOUBLE_ p2 = dbinom_raw(n - x, b, p, q);
	DOUBLE_ p3 = dbinom_raw(n, r + b, p, q);

	return (p1 + p2 - p3); // p1 * p2 / p3;
}

#ifdef	ALTERNATE_DHYPER
/*
 * dhyper
 *
 * From R's dhyper.c.  It is returning logarithm of the density distribution
 */
__device__ DOUBLE_ dhyper2(int x, int r, int b, int n, DOUBLE_ p, DOUBLE_ q) {
	if (n < x || r < x || n - x > b) {
		return -MY_INF; // 0.0;
	}
	if (n == 0) {
		if (x == 0)
			p = 0.0; // 1.0;
		else
			p = -MY_INF; // 0.0;
		return p;
	}
	DOUBLE_ p1 = dbinom_raw(x, r, p, q);
	DOUBLE_ p2 = dbinom_raw(n - x, b, p, q);
	DOUBLE_ p3 = dbinom_raw(n, r + b, p, q);

	return (p1 + p2 - p3); // p1 * p2 / p3;
}
#endif

/*
 * Fisher exact test pvalue computation in GPU
 */
__device__ float fisher_pval_d(int n11, int n12, int n21, int n22) {
	int j, k, lo, hi;
	DOUBLE_ dsum, daux, daux2, dthr, dfin;
	if (n11 < 0 || n12 < 0 || n21 < 0 || n22 < 0)
		return MY_NAN;

	int m = n11 + n21;
	int n = n12 + n22;
	int o = n11 + n12;
	lo = o <= n ? 0 : o - n;
	hi = o < m ? o : m;

#ifdef	ALTERNATE_DHYPER
	DOUBLE_ p = (DOUBLE_) (o) / (DOUBLE_) (m + n);
	DOUBLE_ q = (DOUBLE_) (m + n - o) / (DOUBLE_) (m + n);
	/* dnhyper() */
	for (j = lo, k = 0; j <= hi; ++j, ++k) {
		logdc[k*stride] = dhyper2(j, m, n, o, p, q);
		if (logdc[k*stride] > dmax)
			dmax = logdc[k*stride];
	}
#else
	/* dnhyper() */
	dfin = 0.0;
	dsum = 0.0;
	dthr = dhyper(n11, m, n, o);
	for (j = lo, k = 0; j <= hi; ++j, ++k) {
		daux = dhyper(j, m, n, o);
		daux2 = exp(daux);
		dsum += daux2;
		if (daux <= dthr)
			dfin += daux2;
	}
#endif
	double rex = log10(dfin) - log10(dsum);
	return (float) rex;
}

#undef	ALTERNATE_DHYPER

// Fisher Exact Test P-Value computation kernel
__global__
void k1_fisherpv(IntResultPointers *ptrs, int numRes, int nSamp)
{
	int	resIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
	count_type4	res;
	res.y = ptrs->v[CLASS0][resIdx];
	res.z = ptrs->v[CLASS1][resIdx];
	for (int j = 0; j < NUM_FUNCS; ++j)
	{
		res.w = ptrs->c[CLASS0][resIdx+j*numRes];
		res.x = ptrs->c[CLASS1][resIdx+j*numRes];

//		ptrs->idx[resIdx+j*numRes] = resIdx+j*numRes;
		ptrs->ent[resIdx+j*numRes] = fisher_pval_d(res.y - res.w, res.z - res.x, res.w, res.x);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_KERNEL1_PRINTS)
		if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y < 2)
		{
			printf("\nAt resIdx:%d, FUNC:%d\tc0:%d/%d\tc1:%d/%d, Pv:%f\n", resIdx, j, res.w, res.y, res.x, res.z, ptrs->ent[resIdx+j*numRes]);
		}
#endif
	}
}

#endif	// _FISHER_D_CU_
