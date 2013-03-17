/*
 * kernel2.cu
 *
 *  Created on: 11/01/2012
 *      Author: carlos
 *
 *  Kernel 2: Computation of pairs and coverage for ABK
 *
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

#include <cuda.h>
#include <device_functions.h>

#include "../../inc/gABKEpi.h"
#include "../misc/Timer.h"
#include "../proc_gpu.h"
#include "../misc/print_misc.h"
#include "../ABKEpiGraph.h"
#define	DEBUG_PRINTS
#include "setup2.h"
#include "ancillary2.h"

/*
 * This first kernel will compute the interaction contribution to each sample pair,
 * based on entropy delta of interaction (or self, if single SNP), and alpha or
 * beta cover of the interaction.  In order to dramatically reduce computation time,
 * a staged thresholding scheme will be implemented, with an initial threshold at the
 * entropy level, an alpha or beta cover threshold and finally the running threshold
 * at the pair level.  The running threshold is the worst objective value for the
 * nResPP stored for the pair.
 *
 * Due to limited memory, each block can only compute results for a single interaction.
 * Each thread in a block computes for a SAMPLES_ELEMENT group of pairs, or more:
 *   (nPairs / SAMPLES_ELEMENT) / (threads per block)
 *
 * Basically, blocks are a linear collection of threads (only threadIdx.x changes).
 * In practice, this is no limitation, as we can always convert from a two dimensional
 * block to a single thread index in the block.  It just simplifies index computation
 * a bit.  Block size is only limited by register usage, and hence independent of
 * other problem dimensions.
 */
template <typename T>
__global__ void k2_g(const ABKInputData<T> P, const dim3 firstG, const ABKResultDetails *pr)
{
	const int SAMPLES_ELEMENT = 8 * sizeof(T);
	const int func = BFUNC_AND;
	int idxA = blockIdx.y + firstG.y;	// SNP A goes on the Y direction ("vertical", "row")
	int	idxB = blockIdx.x + firstG.x;
	int intIdx = blockIdx.y * gridDim.x + blockIdx.x;
	int __shared__ *covA;
	int __shared__ *covB;
	int __shared__ myResIndex;
	int __shared__ coverAlpha;
	int __shared__ coverBeta;

	covA = _sm_buffer;
	covB = covA + blockDim.x * blockDim.y;
	covA[threadIdx.x] = 0;
	covB[threadIdx.x] = 0;

	if (idxA > idxB)
		return;

	T	v1b, v2b, e1b, e2b;
	int	sIdx, sIdy;		// Sample index in x and y
	int pairIndex = 0;
	int nEP = P.nELE * (P.nELE + 1) / 2;				// Number of pairs of elements
	int ni = (nEP + blockDim.x - 1) / blockDim.x;		// Number of iterations per thread
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//		printf("B:%d,%d/%d(%dx%d) T:%d,%d/%d(%dx%d) Int:%d nEP:%d  ni:%d\n", blockIdx.x, blockIdx.y, gridDim.x * gridDim.y, gridDim.x, gridDim.y,
//				threadIdx.x, threadIdx.y, blockDim.x*blockDim.y, blockDim.x, blockDim.y, intIdx, nEP, ni);
#endif

	for (int iter = 0; iter < ni; iter++)
	{
		int epi = iter * blockDim.x + threadIdx.x;	// element pair index this thread
		if (epi >= nEP)
			break;
		// Find element row
		int e1, e2;
		ROWCOLATINDEX_V0D(epi,P.nELE,e1,e2);

		v1b = validBitsElement(P.dpt, idxA, idxB, e1, P.numSNPs);
		e1b = binaryFuncElement(P.dpt, idxA, idxB, func, e1, P.numSNPs);
		v2b = validBitsElement(P.dpt, idxA, idxB, e2, P.numSNPs);
		e2b = binaryFuncElement(P.dpt, idxA, idxB, func, e2, P.numSNPs);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//		if (threadIdx.x == 0 && threadIdx.y == 0)
//		{
//			T m11 = maskElem<T>(d_mask1, e1);
//			T m12 = maskElem<T>(d_mask1, e2);
//			printf("[%d,%d:%d]\t%d->%d,%d:\t%lx/%lx(%lx)\t%lx/%lx(%lx), T %d\n", idxA, idxB, intIdx, epi, e1, e2, e1b, v1b, m11, e2b, v2b, m12, threadIdx.x);
//		}
#endif

#ifdef ALTERNATIVE_IMPLEMENTATION
		// An alternative and perhaps faster, without one of the loops, and without flags
		{
			T	equval = (~(e1b ^ e2b));
			T	valval = v1b & v2b;
			T	equ2 = equval >> 1;
			T	valid2 = valval >> 1;
			for(int n = 1; n < SAMPLES_ELEMENT; n++)
			{
				covA[threadIdx.x] += __popcll(equval & equ2 & valval & val2);		// Esto esta mal.  rechequear faltan las mascaras
				covB[threadIdx.x] += __popcll((equval ^ equ2) & valval & val2);		// Esto esta mal.  rechequear faltan las mascaras de clase
				val2 >>= 1;
				equ2 >>= 1;
			}

		}
#endif

		for (int bitgap = 0; bitgap < SAMPLES_ELEMENT; bitgap++)
		{

			T	equval = (~(e1b ^ e2b));
			T	valval = v1b & v2b;
			T	selector = (T) 0x1;
			for (int n = 0; n < SAMPLES_ELEMENT; n++)
			{
				sIdy = e1 * SAMPLES_ELEMENT + n;
				sIdx = e2 * SAMPLES_ELEMENT + (n + bitgap) % SAMPLES_ELEMENT;

				if (sIdx < P.numSamples && sIdy < P.numSamples && sIdx > sIdy)
				{
					pairIndex = INDEXATROWCOL_V00ND(sIdy,sIdx,P.numSamples);
					bool eqeq = ((equval & selector) == selector);
					bool vald = ((valval & selector) == selector);
					covA[threadIdx.x] += (vald && !eqeq && (P.pFlag[pairIndex] & P_ALPHA) == P_ALPHA) ? 1 : 0;
					covB[threadIdx.x] += (vald && eqeq && (P.pFlag[pairIndex] & P_ALPHA) == 0) ? 1 : 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//					if (threadIdx.x == 0 && threadIdx.y == 0)
//					if (threadIdx.x < 3 && intIdx == 1)
//						printf("\t%d\t%d,%d\tr:%d,c:%d  Pi %d  %d[%d] fl:%x, th %d\n", intIdx, e1, e2, sIdy, sIdx, pairIndex, eqeq, vald, P.pFlag[pairIndex], threadIdx.x);
#endif

				}
				selector <<= 1;
			}
			e2b = rotr1(e2b);
			v2b = rotr1(v2b);
		}
	}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//	if (threadIdx.x < 3)
//		printf("Interaction:[%d,%d:%d] cA:%d cB:%d    %dx%d,&%p&%p T%d\n", idxA, idxB, intIdx, covA[threadIdx.x], covB[threadIdx.x],
//				blockDim.x, blockDim.y, &(covA[threadIdx.x]), &(covB[threadIdx.x]), threadIdx.x);
//	printf("ABKResultDetails: %p, CI:%p, wcA %d, wcB %d\n", pr, &(pr->currIndex), P.worstCoverAlpha, P.worstCoverBeta);
#endif

	// TODO: This collection could be done with an atomicAdd()
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		coverAlpha = 0;
		coverBeta = 0;
		int upp = min(blockDim.x * blockDim.y, nEP);
		for (int j = 0; j < upp; j++)
		{
			coverAlpha += covA[j];
			coverBeta += covB[j];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//			printf("%d(%d,%d)   %d   +%d,%d\t+%d,%d\n", j, threadIdx.x, threadIdx.y, intIdx, covA[j], coverAlpha, covB[j], coverBeta);
#endif
		}
//		for (int j = 0; j < upp; j++)
//		{
//			covA[j] = cA;
//			covB[j] = cB;
//		}

		// At this point we have the final covers.  If better than worse then we store it
		if (coverAlpha > P.worstCoverAlpha || coverBeta > P.worstCoverBeta)
		{
			myResIndex = atomicAdd((int *)&(pr->currIndex), 1);
			pr->selected[myResIndex].alphaC = coverAlpha;
			pr->selected[myResIndex].betaC = coverBeta;
			pr->selected[myResIndex].fun = func;
			pr->selected[myResIndex].sA = idxA;
			pr->selected[myResIndex].sB = idxB;
		}
		else
			myResIndex = EMPTY_INDEX_B2;

	}
	__syncthreads();

	// myResIndex, coverAlpha, coverBeta should be available in all threads now.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//	if (threadIdx.x < 4 && threadIdx.y < 4)
//		printf("Interaction:[%d,%d:%d] cA:%d cB:%d rIdx:%d\n", idxA, idxB, intIdx, coverAlpha, coverBeta, myResIndex);
//	__syncthreads();
#endif

	// Finish block if nothing to do
	if (myResIndex < 0 || myResIndex >= pr->maxSelected - 1)
		return;

	covA[threadIdx.x] = 0;
	covB[threadIdx.x] = 0;

	for (int iter = 0; iter < ni; iter++)
	{
		int epi = iter * blockDim.x + threadIdx.x;	// element pair index this thread
		if (epi >= nEP)
			break;
		// Find element row
		int e1, e2;
		ROWCOLATINDEX_V0D(epi,P.nELE,e1,e2);

		v1b = validBitsElement(P.dpt, idxA, idxB, e1, P.numSNPs);
		e1b = binaryFuncElement(P.dpt, idxA, idxB, func, e1, P.numSNPs);
		v2b = validBitsElement(P.dpt, idxA, idxB, e2, P.numSNPs);
		e2b = binaryFuncElement(P.dpt, idxA, idxB, func, e2, P.numSNPs);

		for (int bitgap = 0; bitgap < SAMPLES_ELEMENT; bitgap++)
		{
			T	equval = (~(e1b ^ e2b));
			T	valval = v1b & v2b;
			T	selector = (T) 0x1;
			for (int n = 0; n < SAMPLES_ELEMENT; n++)
			{
				sIdy = e1 * SAMPLES_ELEMENT + n;
				sIdx = e2 * SAMPLES_ELEMENT + (n + bitgap) % SAMPLES_ELEMENT;

				if (sIdx < P.numSamples && sIdy < P.numSamples && sIdx > sIdy)
				{
					pairIndex = INDEXATROWCOL_V00ND(sIdy,sIdx,P.numSamples);
					bool eqeq = ((equval & selector) == selector);
					bool vald = ((valval & selector) == selector);
					bool isAlpha = ((P.pFlag[pairIndex] & P_ALPHA) == P_ALPHA);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//					printf("\t%d\t%d,%d\tr:%d,c:%d  Pi %d  =%dv[%d] fl:%x, th %d a:%d\n", intIdx, e1, e2, sIdy, sIdx, pairIndex,
//							eqeq, vald, P.pFlag[pairIndex], threadIdx.x, a);
#endif
					if (vald && ((!eqeq && isAlpha && coverAlpha > P.worstCovers[pairIndex]) ||
							     (eqeq && !isAlpha && coverBeta > P.worstCovers[pairIndex])))
					{
						unsigned int mmm = EMPTY_INDEX_B4;	// Signals "empty" / unlocked
						// There will be no interference from threads of this block, but there may be from threads of another
						// block.  We use as Id a unique number derived from the blk coordinates
						unsigned int nnw = blockIdx.y * gridDim.x + blockIdx.x;
						unsigned int old = 0;
						while (old != mmm)
						{
							old = atomicCAS(P.locks+pairIndex,mmm,nnw);
						}
						int pilIdx = pairIndex * pr->dResPP;
						bool b = true;
						for (int q = 0; b && q < pr->dResPP; q++)
						{
							if (pr->pairList[pilIdx+q] == EMPTY_INDEX_B2)	// Signals empty
							{
								pr->pairList[pilIdx+q] = myResIndex;
								b = false;
			#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
			//					if (threadIdx.x == 0 && threadIdx.y == 0)
			//						printf("rIdx:%3d pairIdx:%6d  q:%2d\n", myResIndex, pairIndex, q);
			#endif
							}
						}
						// Turn off the lock
						old = atomicExch(P.locks+pairIndex,mmm);
					}
				}
				selector <<= 1;
			}
			e2b = rotr1(e2b);
			v2b = rotr1(v2b);
		}
	}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	__syncthreads();
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//	if (threadIdx.x < 4 && threadIdx.y < 4)
	if (covA[threadIdx.x] > 0)
		printf("T(%d,%d) Interaction:[%d,%d:%d] cA:%d cB:%d rIdx:%d myPairs:%d\n", threadIdx.x, threadIdx.y, idxA, idxB, intIdx,
				coverAlpha, coverBeta, myResIndex, covA[threadIdx.x]);
#endif
}

//################################################################################################################
//################################################################################################################
//################################################################################################################
//################################################################################################################
//################################################################################################################
template <typename T, class Funct>
__device__ void k2_macro(const ABKInputData<T> P, int idxA, int idxB, int boolfunc, const ABKResultDetails *pr, Funct op)
{
	const int SAMPLES_ELEMENT = 8 * sizeof(T);

	if (idxA > idxB)
		return;

	T	v1b, v2b, e1b, e2b;
	int	sIdx, sIdy;		// Sample index in x and y
	int pairIndex = 0;
	const int nEP = P.nELE * (P.nELE + 1) / 2;					// Number of pairs of elements
	const int nIter = (nEP + blockDim.x - 1) / blockDim.x;		// Number of iterations per thread
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//		printf("B:%d,%d/%d(%dx%d) T:%d,%d/%d(%dx%d) Int:%d nEP:%d  ni:%d\n", blockIdx.x, blockIdx.y, gridDim.x * gridDim.y, gridDim.x, gridDim.y,
//				threadIdx.x, threadIdx.y, blockDim.x*blockDim.y, blockDim.x, blockDim.y, intIdx, nEP, ni);
#endif

#pragma unroll
	for (int iter = 0; iter < nIter; iter++)
	{
		int epi = iter * blockDim.x + threadIdx.x;	// element pair index this thread
		if (epi >= nEP)
			break;
		// Find element row
		int e1, e2;
		ROWCOLATINDEX_V0D(epi,P.nELE,e1,e2);

		v1b = validBitsElement(P.dpt, idxA, idxB, e1, P.numSNPs);
		e1b = binaryFuncElement(P.dpt, idxA, idxB, boolfunc, e1, P.numSNPs);
		v2b = validBitsElement(P.dpt, idxA, idxB, e2, P.numSNPs);
		e2b = binaryFuncElement(P.dpt, idxA, idxB, boolfunc, e2, P.numSNPs);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//		if (threadIdx.x == 0 && threadIdx.y == 0)
//		{
//			T m11 = maskElem<T>(d_mask1, e1);
//			T m12 = maskElem<T>(d_mask1, e2);
//			printf("[%d,%d:%d]\t%d->%d,%d:\t%lx/%lx(%lx)\t%lx/%lx(%lx), T %d\n", idxA, idxB, intIdx, epi, e1, e2, e1b, v1b, m11, e2b, v2b, m12, threadIdx.x);
//		}
#endif

#pragma unroll
		for (int bitgap = 0; bitgap < SAMPLES_ELEMENT; bitgap++)
		{

			T	equval = (~(e1b ^ e2b));
			T	valval = v1b & v2b;
			T	selector = (T) 0x1;
			for (int n = 0; n < SAMPLES_ELEMENT; n++)
			{
				sIdy = e1 * SAMPLES_ELEMENT + n;
				sIdx = e2 * SAMPLES_ELEMENT + (n + bitgap) % SAMPLES_ELEMENT;

				if (sIdx < P.numSamples && sIdy < P.numSamples && sIdx > sIdy)
				{
					pairIndex = INDEXATROWCOL_V00ND(sIdy,sIdx,P.numSamples);
					bool eqeq = ((equval & selector) == selector);
					bool vald = ((valval & selector) == selector);

					//////////////// Functor
					op(vald, eqeq, ((P.pFlag[pairIndex] & P_ALPHA) == P_ALPHA), pairIndex, P);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//					if (threadIdx.x == 0 && threadIdx.y == 0)
//						printf("\t%d\t%d,%d\tr:%d,c:%d  Pi %d  %d[%d] fl:%x, th %d\n", intIdx, e1, e2, sIdy, sIdx, pairIndex, eqeq, vald, P.pFlag[pairIndex], threadIdx.x);
#endif
				}
				selector <<= 1;
			}
			e2b = rotr1(e2b);
			v2b = rotr1(v2b);
		}
	}
}


template<typename T>
class CoverCalc
{
	int *cover;
public:
	__device__ CoverCalc(int *ptr): cover(ptr)
	{
		cover[0] = 0;
		cover[1] = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//		printf("th(%d,%d)\tbl(%d,%d)\t%p %p %p\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, cover, &cover[0], &cover[1]);
#endif
	}
	__device__ void operator() (bool va, bool eq, bool isAlpha, int pairIndex, const ABKInputData<T> P) const
	{
		cover[0] += (va && (!eq) && isAlpha) ? 1 : 0;
		cover[1] += (va && eq && (!isAlpha)) ? 1 : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//		printf("th(%d,%d)\tbl(%d,%d)\tpi:%d %d %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, pairIndex, cover[0], cover[1]);
#endif
	}
};

template<typename T>
class BestStorer
{
	const ABKResultDetails *_pr;
	const int3  *res;
public:
	__device__ BestStorer(const ABKResultDetails *pr, const int3 *rs): _pr(pr), res(rs)
	{
	}
	__device__ void operator() (bool vald, bool equ, bool isAlpha, int pairIndex, const ABKInputData<T> P) const
	{
		if (vald && ((!equ && isAlpha && res->x > P.worstCovers[pairIndex]) ||
				     (equ && !isAlpha && res->y > P.worstCovers[pairIndex])))
		{
			unsigned int mmm = EMPTY_INDEX_B4;	// Signals "empty" / unlocked
			// There will be no interference from threads of this block, but there may be from threads of another
			// block.  We use as Id a unique number derived from the blk coordinates
			unsigned int nnw = blockIdx.y * gridDim.x + blockIdx.x;
			unsigned int old = 0;
			while (old != mmm)
			{
				old = atomicCAS(P.locks+pairIndex,mmm,nnw);
			}
			int pilIdx = pairIndex * _pr->dResPP;
			bool b = true;
			for (int q = 0; b && q < _pr->dResPP; q++)
			{
				if (_pr->pairList[pilIdx+q] == EMPTY_INDEX_B2)	// Signals empty
				{
					_pr->pairList[pilIdx+q] = res->z;
					b = false;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//					if (threadIdx.x == 0 && threadIdx.y == 0)
//						printf("rIdx:%3d pairIdx:%6d  q:%2d\n", myResIndex, pairIndex, q);
#endif
				}
			}
			// Turn off the lock
			old = atomicExch(P.locks+pairIndex,mmm);
		}
	}
};

__device__ int2 collectCover(const int nEP, const int intIdx)
{
	int __shared__ *scover;
	scover = _sm_buffer;
	int2 	daCover;		// x is Alpha, y is Beta
	daCover.x = 0;
	daCover.y = 0;
	const int upp = min(blockDim.x * blockDim.y, nEP);
	for (int j = 0; j < upp; j++)
	{
		daCover.x += scover[2*j];
		daCover.y += scover[2*j+1];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//		printf("%d(%d,%d)   intIdx:%d   +%d,%d\t+%d,%d   %p\n", j, threadIdx.x, threadIdx.y, intIdx, scover[2*j], daCover.x, scover[2*j+1], daCover.y, scover + 2*j);
#endif
	}
	return daCover;
}

template <typename T>
__global__ void k2_v2(const ABKInputData<T> P, const dim3 firstG, const ABKResultDetails *pr)
{
	const int func = BFUNC_AND;
	int intIdx = blockIdx.y * gridDim.x + blockIdx.x;
	int idxA = blockIdx.y + firstG.y;	// SNP A goes on the Y direction ("vertical", "row")
	int	idxB = blockIdx.x + firstG.x;
	int3 __shared__ *res;
	if (idxA > idxB)
		return;

	k2_macro(P, idxA, idxB, func, pr, CoverCalc<T>((int *)_sm_buffer + 2*(threadIdx.y*blockDim.x+threadIdx.x)));
	// TODO: Manage shared mem locally and not via external buffer
	res = (int3 *)((int *) _sm_buffer + 2*(blockDim.x * blockDim.y));
	__threadfence_block();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int2	r = collectCover(P.nELE * (P.nELE + 1) / 2, intIdx);
		PILindex_t	myResIndex;
		if (r.x > P.worstCoverAlpha || r.y > P.worstCoverBeta)
		{
			myResIndex = atomicAdd((int *)&(pr->currIndex), 1);
			pr->selected[myResIndex].alphaC = r.x;
			pr->selected[myResIndex].betaC = r.y;
			pr->selected[myResIndex].fun = func;
			pr->selected[myResIndex].sA = idxA;
			pr->selected[myResIndex].sB = idxB;
		}
		else
			myResIndex = EMPTY_INDEX_B2;
		res->x = r.x;
		res->y = r.y;
		res->z = myResIndex;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
		printf("Interaction:%d[%d,%d] cA:%d cB:%d selIdx:%d\n", intIdx, idxA, idxB, res->x, res->y, res->z);
//		printf("Interaction:%d[%d,%d] cA:%d cB:%d selIdx:%d\n", intIdx, idxA, idxB, r.x, r.y, myResIndex);
#endif
	}
	__threadfence_block();
	// Finish block if nothing to do  // Next func !!
	if (res->z == EMPTY_INDEX_B2 || res->z >= pr->maxSelected - 1)
		return;

	k2_macro(P, idxA, idxB, func, pr, BestStorer<T>(pr, res));

}

// Launcher of the kernel 2 function.
template <typename T, typename Key>
void launch_process_gpu_2(PlinkReader<T> *pr, ABKEpiGraph<Key, T> &abkeg, const struct _paramP1 &par)
{
	ABKInputData<T>		d_InDPT;
	size_t	byteCounter;
	double	totTime = 0.0f;

	Timer tt;
	CUDACHECK(cudaSetDevice(par.gpuNum),par.gpuNum);
	byteCounter = allocSendData_device(*pr, abkeg, d_InDPT, par);
	clog << "K2: InputData: " << byteCounter << std::endl;

	ABKResultDetails	ld_abkr,	// Local copy of device pointers
						*p_abkr,	// Pointer to device struct
						h_abkr;		// Host struct
	if (!computeResultBufferSize(ld_abkr, abkeg, par))
	{
		freeData_device(d_InDPT, par);
		return;
	}
	h_abkr = ld_abkr;

	byteCounter += allocResults_device(ld_abkr, &p_abkr, par);
	byteCounter += allocResults_host(h_abkr, par);
	size_t availMem, totalMem;
	CUDACHECK(cudaMemGetInfo(&availMem,&totalMem),par.gpuNum);
	totTime += tt.stop();

	clog << "K2: Alloc time " << totTime << ", Available mem < " << (availMem+ONEMEGA-1)/ONEMEGA << "MB" << endl;

	// Copy class masks to constant memory
	if (pr->elementsSNP() > CONST_MASK_MAX_SIZE)
	{
		cerr << "K2: Class Masks do not fit in constant memory.  Change compilation size" << endl;
		return;
	}
	CUDACHECK(cudaMemcpyToSymbol(d_mask0,pr->classData(1)->mask(),pr->elementsSNP()*sizeof(T),0,cudaMemcpyHostToDevice),par.gpuNum);
	CUDACHECK(cudaMemcpyToSymbol(d_mask1,pr->classData(2)->mask(),pr->elementsSNP()*sizeof(T),0,cudaMemcpyHostToDevice),par.gpuNum);

#define BLOCK_DIM	32
	clog << endl;
	cout << endl;
	dim3	block(BLOCK_DIM,1);
//	dim3	grid(pr->numSNPs(), pr->numSNPs());
	dim3	grid(4,4);
	dim3	fgr(0,0,0);

	initCycleResults_device<T,Key>(abkeg, d_InDPT, ld_abkr, p_abkr, par);
	clog << endl;
	sendWorstCovers(abkeg,d_InDPT,par);
	clog << endl;
	k2_g<T><<<grid,block,16384>>>(d_InDPT, fgr, p_abkr);
//	k2_v2<T><<<grid,block,16384>>>(d_InDPT, fgr, p_abkr);
	CUDACHECK(cudaDeviceSynchronize(),par.gpuNum);
	transferBack(p_abkr, &h_abkr, par);
	mergeResults(abkeg,&h_abkr,par);
	printResultBuffer(&h_abkr, abkeg.getPairFlags(), abkeg.getWorstCovers());
	abkeg.print();

#ifdef CACA
	{
		// We want to compute nSNPs * nSNPs interactions in blocks of dimension BLOCK_DIM x BLOCK_DIM, considering
		// only the lower triangular part of the interaction matrix is computed
		size_t nBside = (pr->numSNPs() + BLOCK_DIM - 1) / BLOCK_DIM;
		size_t nTotBlocks = nBside * (nBside + 1) / 2;
		int s4nir = (int) (log2((float) nIRmax) / 2.0f);
		s4nir = (int) exp2((float) s4nir);
		int b4nir = s4nir / BLOCK_DIM;
		fprintf(stderr, "K1: Block(%d,%d) th/B:%d B/side:%lu, totB:%lu S4nIR:%d B4nIR:%d\n",
				BLOCK_DIM, BLOCK_DIM, (BLOCK_DIM*BLOCK_DIM), nBside, nTotBlocks, s4nir, b4nir);
		dim3	block(BLOCK_DIM,BLOCK_DIM);
		dim3	grid(b4nir,b4nir);
		dim3	fgr(0,0,0);

		const unsigned int nny = grid.y * block.y;
		const unsigned int nnx = grid.x * block.x;
		for (fgr.y = 0; fgr.y < pr->numSNPs() - nny; fgr.y += nny)
		{
			for (fgr.x = 0; fgr.x < fgr.y; fgr.x += nnx)
			{

				clog << "K1: Launch <<(" << grid.x << "," << grid.y << "," << grid.z << "),("
						<< block.x << "," << block.y << "," << block.z << ")>>\t fgr="
						<< fgr.x << "," << fgr.y <<  "\t ent[numTopmost]: " << l_tops[nIRmax].ent
						<< "\tT:" << tt.stop() << endl;
				k2_g<T><<<grid,block>>>(d_bitA, d_bita, d_valid, d_tops, nIRmax, fgr, pr->numSNPs(), pr->numSamples(), pr->elementsSNP());

#ifdef	CHECK_LOOP_LIMITS
				// WARN: This function destroys the content of l_tops !!
				checkLaunchLoops(grid, block, nIRmax, fgr, l_tops, pr->numSNPs(), pr->numSamples(), pr->elementsSNP());
#else
				// TODO try sorting on the gpu (once this is working...
				qsort(l_tops, (numTopmost + nIRmax), sizeof(InteractionResult), compar_result);
//				printResults(l_tops, numTopmost+nIRmax, 100000);
#endif

				CUDACHECK(cudaDeviceSynchronize(),par.gpuNum);
				CUDACHECK(cudaMemcpy(l_tops,d_tops,nIRmax*sizeof(InteractionResult),cudaMemcpyDeviceToHost),par.gpuNum);
				CUDACHECK(cudaDeviceSynchronize(),par.gpuNum);
			}
		}
	}

	qsort(l_tops, (numTopmost + nIRmax), sizeof(InteractionResult), compar_result);
//	printResults(l_tops, numTopmost+nIRmax, 100000);

	double mt = tt.stop();
	clog << "K1: Copy+Run time " << mt << " Total: " << mt + totTime << endl;

	// Final out copy, reverse
	for (size_t j = 0; j < numTopmost; j++)
	{
		tops[j] = l_tops[numTopmost + nIRmax - j - 1];
	}
	printResults(tops, numTopmost, 1000);

#endif

	freeResults_host(h_abkr, par);
	freeResults_device(ld_abkr, p_abkr, par);
	freeData_device(d_InDPT, par);

	return;
}

void process_gpu_2(PlinkReader<ui8> *pr, ABKEpiGraph<int64_t, ui8> &abkeg, const struct _paramP1 &par)
{
	launch_process_gpu_2(pr, abkeg, par);
}

__host__ void process_gpu_2(PlinkReader<ui8> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ptops, const struct _paramP1 &par)
{
//	launch_process_gpu_2<ui8>(pr, nPairs, nResPP, ptops, par);
}

__host__ void process_gpu_2(PlinkReader<ui4> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ptops, const struct _paramP1 &par)
{

}
