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
#include <math.h>

#ifndef M_2PI
#define M_2PI           6.283185307179586476925286766559        /* 2*pi */
#endif
#ifndef	NAN
#define	NAN	FP_NAN
#endif
#ifndef	INFINITY
#define	INFINITY	FP_INFINITE
#endif

static double stirlerr(int n);
static double dhyper(int x, int r, int b, int n);
static double dbinom_raw(int x, int n, double p, double q);
static double bd0(double x, double np);
static double stirlerr(int n);

float fisher_pval(int n11, int n12, int n21, int n22) {
	int j;
	double dsumd, dsumn, dthr, daux, daux2;
	if (n11 < 0 || n12 < 0 || n21 < 0 || n22 < 0)
		return NAN;

	int m = n11 + n21;
	int n = n12 + n22;
	int o = n11 + n12;
	int lo = o <= n ? 0 : o - n;
	int hi = o < m ? o : m;

	dsumd = 0.0;
	dsumn = 0.0;
	dthr = dhyper(n11, m, n, o);
	for (j = lo; j <= hi; ++j)
	{
		daux = dhyper(j, m, n, o);
		daux2 = exp(daux);
		dsumd += daux2;
		if (daux <= dthr)
			dsumn += daux2;
	}
	return (float)(log10(dsumn) - log10(dsumd));
}

/*
 * dhyper
 *
 * From R's dhyper.c.  It is returning logarithm of the density distribution
 */
static double dhyper(int x, int r, int b, int n) {
	if (n < x || r < x || n - x > b)
		return -INFINITY; // 0.0;
	if (n == 0) {
		if (x == 0)
			return 0.0; // 1.0;
		else
			return -INFINITY; // 0.0;
	}
	double p = (double) n / (double) (r + b);
	double q = (double) (r + b - n) / (double) (r + b);

	double p1 = dbinom_raw(x, r, p, q);
	double p2 = dbinom_raw(n - x, b, p, q);
	double p3 = dbinom_raw(n, r + b, p, q);

	return (p1 + p2 - p3); // p1 * p2 / p3;
}

/*
 * dbinom_raw
 *
 * From R's dbinom.c.  Returns only log of values.
 */
static double dbinom_raw(int x, int n, double p, double q) {
	double lf, lc;

	if (p == 0) {
		if (x == 0)
			return 0.0; // : 1.0;
		else
			return -INFINITY; // : 0.0;
	}
	if (q == 0) {
		if (x == 0)
			return 0.0; // : 1.0;
		else
			return -INFINITY; // : 0.0;
	}

	if (x == 0) {
		if (n == 0)
			return 0.0; // : 1.0;
		lc = (p < 0.1) ? -bd0((double) n, (double) n * q) - (double) n * p : (double) n * log(q);
		return lc; // : exp(lc);
	}
	if (x == n) {
		lc = (q < 0.1) ? -bd0((double) n, (double) n * p) - (double) n * q : (double) n * log(p);
		return lc; // : exp(lc);
	}
	if (x < 0 || x > n)
		return -INFINITY; // : 0.0;

	/* n*p or n*q can underflow to zero if n and p or q are small.  This
	 used to occur in dbeta, and gives NaN as from R 2.3.0.  */
	lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0((double) x, (double) n * p) - bd0((double) (n - x), (double) n * q);

	/* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
	/* Upto R 2.7.1:
	 * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
	 * -- following is much better for  x << n : */
	lf = log(M_2PI) + log((double) x) + log1p(-(double) x / (double) n);

	lc -= 0.5 * lf;
	return (lc); // : exp(lc);
}

/*
 * bd0
 *
 * From R's bd0.c
 */
double bd0(double x, double np) {
	double ej, s, s1, v;
	int j;

	if (isinf(x) || isinf(np) || np == 0.0)
		return NAN;

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

static double stirlerr(int n) {

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

	/*
	 error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
	 */
	const static double sferr[16] = { 0.0, /* n=0 - wrong, place holder only */
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

	if (n <= 15) {
		return (sferr[n]);
	}

	double nn = (double) n * (double) n;
	if (n > 500)
		return ((S0 - S1 / nn) / (double) n);
	if (n > 80)
		return ((S0 - (S1 - S2 / nn) / nn) / (double) n);
	if (n > 35)
		return ((S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / (double) n);
	/* 15 < n <= 35 : */
	return ((S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / (double) n);
}

