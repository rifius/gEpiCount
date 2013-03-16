/*
 * print_misc.cpp
 *
 *  Created on: 26/01/2012
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
#ifndef	PRINT_MISC_H
#define	PRINT_MISC_H

#include "../../inc/gEpiCount.h"
#include "../proc_gpu.h"

#define		PRINT_STRIDE	100000		// For debug, stride in printing results
#define		PRINT_CONTEXT	5

static const string _binaryFuncName[BFUNC_LAST + 1] = { "AND", "NAND", "ANDN", "NANDN", "XOR", "Invalid" };

// Auxiliary function to print results
inline void printResults(InteractionResult *tops, int numT, int stride = 1) {
	for (int j = 0; j < numT; j += stride)
		fprintf(stderr, "%d\tsID(%d,%d)\t%d,%d,%d,%d\t%g\n", j, tops[j].sA, tops[j].sB, tops[j].n.w,
				tops[j].n.x, tops[j].n.y, tops[j].n.z, tops[j].ent);
}

// Auxiliary function to print results.  GPU data
__host__ inline void printResults(const IntResultPointers &h_ptrs, int numT, dim3 blk, dim3 grd,
		dim3 off, int stride = 1) {
	for (int j = 0; j < numT; j += stride) {
		int ki = h_ptrs.idx[j];
		int sA = SNPA_FROM_RINDEX_(ki / NUM_FUNCS, blk, grd, off);
		int sB = SNPB_FROM_RINDEX_(ki / NUM_FUNCS, blk, grd, off);
		fprintf(stderr, "%d\tsID(%d,%d)\t%d,%d,%d,%d\t%g\t%d\n", j, sA, sB, h_ptrs.c[CLASS0][ki],
				h_ptrs.c[CLASS1][ki], h_ptrs.v[CLASS0][ki], h_ptrs.v[CLASS1][ki], h_ptrs.ent[j],
				h_ptrs.idx[j]);
	}
}

__host__ inline void printResultBuffer(const ABKResultDetails *pr, const unsigned char *flags, const int *worstCs)
{
	fprintf(stderr,"IR\tidxA\tidxB\tfun\tcovA\tcovB\n");
	for (int j = 0; j < pr->currIndex; j++)
	{
		PairInteractionResult *ir = pr->selected + j;
		fprintf(stderr, "%d\t%d\t%d\t%s\t%d\t%d\n", j, ir->sA, ir->sB, _binaryFuncName[ir->fun].c_str(), ir->alphaC, ir->betaC);
	}
	// This is just to check the cover numbers
	fprintf(stderr,"\n");
	for (int j = 0; j < pr->nPairs; j++)
	{
		int base = j * pr->dResPP;
		char v = (flags[j] & P_VALID) == P_VALID ? '.': 'I';
		char a = (flags[j] & P_ALPHA) == P_ALPHA ? 'A': 'B';
		fprintf(stderr, "pair:%d.%c%c %02x\twc:%d", j, a, v, flags[j], worstCs[j]);
		for (int k = 0; k < pr->dResPP; k++)
		{
			if (pr->pairList[base + k] != EMPTY_INDEX_B2)
			{
				fprintf(stderr, "\t%d", pr->pairList[base + k]);
				if (v != 'I')
				{
					if (a == 'A')
						pr->selected[pr->pairList[base+k]].alphaC--;
					else
						pr->selected[pr->pairList[base+k]].betaC--;
				}
			}
			else
			{
				fprintf(stderr,"\t");
			}
		}
		fprintf(stderr,"\n");
	}
	fprintf(stderr,"\n");
	for (int j = 0; j < pr->currIndex; j++)
	{
		PairInteractionResult *ir = pr->selected + j;
		if (ir->alphaC != 0 || ir->betaC != 0)
			fprintf(stderr, "*%d\t%d\t%d\t%d\t%d\n", j, ir->sA, ir->sB, ir->alphaC, ir->betaC);
	}
}

// Aux function to print interaction result
template<typename T>
inline void printInteractions(const PlinkReader<T> *pr, const InteractionResult *tops, int numT,
		int stride = 1, int context = 0) {
	fprintf(stderr, "\nRank:\tidxA[id allele chr:pos]\tFunc\tidxB[id allele chr:pos]\tObj\tFPv\t(c00,c01,c10,c11)\n");
	if (stride - 2 * context < 1) context = stride / 2;
	for (int j = 0; j < numT; j += stride) {
		for (int k = j - context < 0 ? 0 : j - context;
				k < (j + context + 1 > numT ? numT : j + context + 1); ++k)
			printInteraction(k, pr, tops[k]);
	}
}

// Aux function to print interaction result
template<typename T>
inline void printInteraction(int num, const PlinkReader<T> *pr, const InteractionResult &ir) {
	SNP A = pr->getSnp(ir.sA);
	SNP B = pr->getSnp(ir.sB);
	float pv = fisher_pval(ir.n.w, ir.n.y - ir.n.w, ir.n.x, ir.n.z - ir.n.x);
	fprintf(stderr, "%d:\t%d[%s %s %d:%d]\t%s\t%d[%s %s %d:%d]\t%g\t%g\t(%d,%d,%d,%d)\n", num,
			ir.sA, A.Id().c_str(), A.Alleles().c_str(), A.Chr(), A.Pos(),
			_binaryFuncName[(int) ir.fun].c_str(), ir.sB, B.Id().c_str(), B.Alleles().c_str(),
			B.Chr(), B.Pos(), ir.ent, pv, ir.n.w, ir.n.y - ir.n.w, ir.n.x, ir.n.z - ir.n.x);
}

//================================ Print data functions used in main or readers
// Prints ndata SNP data at row koff
inline void print_data(const PlinkReader<ui8> *pr, int koff = 0, int ndata = 1) {
	int ese = pr->elementsSNP();
	int off = koff * ese;
	printf("\n---- Data %d", off);
	for (int j = 0; j < ndata; ++j) {
		printf("\n%d:\n", koff + j);
		for (int k = 0; k < ese; ++k) {
			printf("%016Lx ", pr->BitData()[k + off + j * ese]);
			if (k % 4 == 3) printf("\n");
		}
	}
	printf("\n---- Valid");
	for (int j = 0; j < ndata; ++j) {
		printf("\n%d:\n", koff + j);
		for (int k = 0; k < ese; ++k) {
			printf("%016Lx ", pr->ValidData()[k + off + j * ese]);
			if (k % 4 == 3) printf("\n");
		}
	}
	printf("\n---- Mask0\n");
	for (int k = 0; k < ese; k++) {
		printf("%016Lx ", pr->classData(1)->mask()[k]);
		if (k % 4 == 3) printf("\n");
	}

	printf("\n---- Mask1\n");
	for (int k = 0; k < ese; k++) {
		printf("%016Lx ", pr->classData(2)->mask()[k]);
		if (k % 4 == 3) printf("\n");
	}

	printf("\n");
}

inline void print_data(const PlinkReader<ui4> *pr, int koff = 0, int ndata = 1) {
	int ese = pr->elementsSNP();
	int off = koff * ese;
	printf("\n---- Data %d", off);
	for (int j = 0; j < ndata; ++j) {
		printf("\n%d:\n", koff + j);
		for (int k = 0; k < ese; k += 2) {
			printf("%08x%08x ", pr->BitData()[k + off + j * ese + 1],
					pr->BitData()[k + off + j * ese]);
			if (k % 8 >= 6) printf("\n");
		}
	}
	printf("\n---- Valid");
	for (int j = 0; j < ndata; ++j) {
		printf("\n%d:\n", koff + j);
		for (int k = 0; k < ese; k += 2) {
			printf("%08x%08x ", pr->ValidData()[k + off + j * ese + 1],
					pr->ValidData()[k + off + j * ese]);
			if (k % 8 >= 6) printf("\n");
		}
	}
	printf("\n---- Mask0\n");
	for (int k = 0; k < ese; k += 2) {
		printf("%08x%08x ", pr->classData(1)->mask()[k + 1], pr->classData(1)->mask()[k]);
		if (k % 8 >= 6) printf("\n");
	}
	printf("\n---- Mask1\n");
	for (int k = 0; k < ese; k += 2) {
		printf("%08x%08x ", pr->classData(2)->mask()[k + 1], pr->classData(2)->mask()[k]);
		if (k % 8 >= 6) printf("\n");
	}

	printf("\n");
}


#endif	// PRINT_MISC_H
