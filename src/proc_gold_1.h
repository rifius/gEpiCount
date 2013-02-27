/*
 * proc_gold_1.h
 *
 *  Created on: 30/12/2011
 *      Author: carlos
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

#ifndef PROC_GOLD_1_H_
#define PROC_GOLD_1_H_

#include "proc_gold.h"
#include "kernels/entropy_inline.h"

#include <thrust/sort.h>
#include <thrust/host_vector.h>

// Store without updating worst
#define		STORE_RESULT(r,e,j,k,f,w,t)			{	w.index++; t[w.index].ent = e; t[w.index].n = r; \
													t[w.index].sA = j; t[w.index].sB = k; t[w.index].fun = f; }
// Store and update worst value
#define		STORE_RESULT_U(r,e,j,k,f,w,t)		{	w.val = e; STORE_RESULT(r,e,j,k,f,w,t); }

// From smallest (most negative) to largest (most positive)
// Functor for qsort()
static int compar_result(const void *a, const void *b)
{
	if (((InteractionResult*) b)->ent - ((InteractionResult*) a)->ent < 0.0)
		return 1;
	else
		return -1;
}

// Functor for thrust::sort
struct IRCompare
{
	bool operator()(const InteractionResult &x, const InteractionResult &y) const
	{
		return x.ent < y.ent;
	}
};

struct _worst
{
	float val;
	int index;
};

/////////////////////
// First Kernel.  Computation of binary function and topmost list.
/////////////////////
template <typename T>
void process_gold_1(PlinkReader<T> *pr, InteractionResult *tops, const struct _paramP1 &par)
{
	_do_preload();
	const size_t nSNPs = pr->numSNPs();
	const size_t nELEs = pr->elementsSNP();
	struct _worst worst;
	worst.val = numeric_limits<float>::max();
	worst.index = -1;

	T *val0 = pr->classData(1)->mask();
	T *val1 = pr->classData(2)->mask();
	if (val0 == NULL || val1 == NULL)
	{
		cerr << "Error in class mask data" << std::endl;
		return;
	}

	// We manage internal buffer for tops, so we allocate outside only what is needed.
	InteractionResult *itops = new InteractionResult[2 * par.topmost];
	std::memset(itops, 0, 2 * par.topmost * sizeof(InteractionResult));

	T *aMask0 = new T[nELEs];
	std::memset(aMask0, 0, nELEs * sizeof(T));

	T *aMask1 = new T[nELEs];
	std::memset(aMask1, 0, nELEs * sizeof(T));

	double totTime = 0.0f;
	Timer tt, te;
	for (size_t j = 0; j < nSNPs; j++)
	{
		size_t rowOff = j * nELEs;
		T *snpA = pr->BitData() + rowOff;
		T *valA = pr->ValidData() + rowOff;
		int4 res = {0};
		for (size_t k = 0; k <= j; k++)
		{
			size_t colOff = k * nELEs;
			T *snpB = pr->BitData() + colOff;
			T *valB = pr->ValidData() + colOff;
			for (size_t m = 0; m < nELEs; m++)
			{
				aMask0[m] = val0[m] & valA[m] & valB[m];
				aMask1[m] = val1[m] & valA[m] & valB[m];
			}
			for (int fun = 0; fun < NUM_FUNCS; ++fun)
			{
				res = binFunc(snpA, snpB, aMask0, aMask1, fun, nELEs);
				float ee = entropyDelta4(res);
				// We look for the largest decrease in entropy, that is, the most negative deltaE
				if (ee < worst.val)
				{
					if (worst.index < 2*par.topmost-1)
					{
						STORE_RESULT(res,ee,j,k,fun,worst,itops);
					}
					else
					{
						thrust::sort(itops, itops+2*par.topmost, IRCompare());
						worst.index = par.topmost-1;
						worst.val = itops[worst.index].ent;
						clog << "G1: Sorting... Limit: " << worst.val << std::endl;
						if (ee < worst.val)
						{
							STORE_RESULT(res,ee,j,k,fun,worst,itops);
						}
					}
				}
			}
		}
		// This print is adjusted so that it comes at the same position as in gpu (check grid and block sizes).
		if ((j+1)%1024 == 0)
		{
			totTime = tt.stop();
			float nt = (float) pr->numSNPs() / (float) (j+1);		// Fraction so far is 1/nt
			nt = nt * nt;
			float rem = (nt - 1.0f) * totTime;
			float pct = 100.0f / nt;
			string uu = "sec";
			if (rem > 3600.0)
			{
				rem /= 3600.0;
				uu = "hr";
			}
			clog << "G1: Total time " << totTime << "sec (" << pct << "%)" << " Remaining: " << rem << uu << endl;
			if (par.intermediate)
			{
				// We have to sort before print to get latest results in
				thrust::sort(itops, itops+2*par.topmost, IRCompare());
				worst.index = par.topmost-1;
				worst.val = itops[worst.index].ent;
				clog << "G1: Sorting... Limit: " << worst.val << std::endl;
				clog << "G1: Last processed: " << j << "," << j << endl;
				printInteractions(pr, itops, par.topmost, PRINT_STRIDE, PRINT_CONTEXT);
			}
		}
	}
	// Sort before last copy
	thrust::sort(itops, itops+2*par.topmost, IRCompare());
	worst.index = par.topmost-1;
	worst.val = itops[worst.index].ent;
	clog << "G1: Sorting... Limit: " << worst.val << std::endl;
	totTime = tt.stop();
	std::memcpy(tops, itops, par.topmost * sizeof(InteractionResult));
	clog << "process_gold_1: " << totTime << " " << std::endl;

	delete[] aMask0;
	delete[] aMask1;
	delete[] itops;
}

#endif /* PROC_GOLD_1_H_ */
