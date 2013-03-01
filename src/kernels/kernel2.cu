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

#include "../../inc/gABKEpi.h"
#include "../misc/Timer.h"
#include "../proc_gpu.h"
#define	DEBUG_PRINTS
#include "setup2.h"

template <typename T>
__forceinline__ __device__ T validBitsElement(const struct _dataPointersPairsD<T> P, int idxA, int idxB, int ele, int nSNPs)
{
	T	res = 0;
	if (idxA >= nSNPs || idxB >= nSNPs)
		return res;
	T		*vA = P.dpt.valid + idxA;
	T		*vB = P.dpt.valid + idxB;
	int idxj = ele * nSNPs;
	res = vA[idxj] & vB[idxj];								// Valid bits
	res &= (maskElem<T>(d_mask0, ele) | maskElem<T>(d_mask1, ele));
	return res;
}

template <typename T>
__forceinline__ __device__ T binaryFuncElement(const struct _dataPointersPairsD<T> P, int idxA, int idxB, int fun, int ele, int nSNPs)
{
	T	res = 0;
	if (idxA >= nSNPs || idxB >= nSNPs)
		return res;
	T		*bA = P.dpt.bits + idxA;
	T		*bB = P.dpt.bits + idxB;

	int idxj = ele * nSNPs;

	switch(fun)
	{
	case BFUNC_AND:
		res = (bA[idxj] & bB[idxj]);
//		cc[CLASS0][BFUNC_AND] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_AND] += __popcll(c & v & m1);
		break;
	case BFUNC_NAND:
		res = ((~bA[idxj]) & bB[idxj]);
//		cc[CLASS0][BFUNC_NAND] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_NAND] += __popcll(c & v & m1);
		break;
	case BFUNC_ANDN:
		res = (bA[idxj] & (~bB[idxj]));
//		cc[CLASS0][BFUNC_ANDN] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_ANDN] += __popcll(c & v & m1);
		break;
	case BFUNC_NANDN:	// (implemented as ~(a | b)
		res = (~(bA[idxj] | bB[idxj]));
//		cc[CLASS0][BFUNC_NANDN] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_NANDN] += __popcll(c & v & m1);
		break;
	case BFUNC_XOR:
		res = (bA[idxj] ^ bB[idxj]);
//		cc[CLASS0][BFUNC_XOR] += __popcll(c & v & m0);
//		cc[CLASS1][BFUNC_XOR] += __popcll(c & v & m1);
		break;
	default:
		break;
	}
	return res;
}

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
__global__ void k2_g(const struct _dataPointersPairsD<T> P, const dim3 firstG, const size_t nSNPs, const size_t nSamp,
		const size_t nELE, const size_t nPairs, const int nResPP, PairInteractionResult *ptops)
{
	const int SAMPLES_ELEMENT = 8 * sizeof(T);
	const int func = BFUNC_AND;
	int idxA = blockIdx.y + firstG.y;	// SNP A goes on the Y direction ("vertical", "row")
	int	idxB = blockIdx.x + firstG.x;
	int intIdx = blockIdx.y * gridDim.x + blockIdx.x;
	int __shared__ *covA;
	int __shared__ *covB;

	covA = _sm_buffer;
	covB = covA + blockDim.x * blockDim.y;
	covA[threadIdx.x] = 0;
	covB[threadIdx.x] = 0;

	if (idxA > idxB)
		return;

	T	c, v1b, v2b, e1b, e2b;
	int	sIdx, sIdy;		// Sample index in x and y
	int pairIndex = 0;
	int nEP = nELE * (nELE + 1) / 2;				// Number of pairs of elements
	int ni = (nEP + blockDim.x - 1) / blockDim.x;	// Number of iterations per thread
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
#ifdef ALTERNATIVE_IMPLEMENTATION
		e1 = 2*nELE - 1;
		{
			float a = (float) e1 * (float) e1 - 8.0 * (float) (epi - nELE + 1);
			a = ((float) e1 - sqrtf(a)) / 2.0;
			e1 = (int) ceilf(a);
		}
		// and element column
		e2 = epi - e1 * (nELE - 1) + (e1 * (e1 - 1)) / 2;
#else
		ROWCOLATINDEX_V0D(epi,nELE,e1,e2);
#endif

		v1b = validBitsElement(P, idxA, idxB, e1, nSNPs);
		e1b = binaryFuncElement(P, idxA, idxB, func, e1, nSNPs);
		v2b = validBitsElement(P, idxA, idxB, e2, nSNPs);
		e2b = binaryFuncElement(P, idxA, idxB, func, e2, nSNPs);

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
			for(int n = 1; n < SAMPLESAMPLES_ELEMENT; n++)
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

				if (sIdx < nSamp && sIdy < nSamp && sIdx > sIdy)
				{
					pairIndex = INDEXATROWCOL_V00ND(sIdy,sIdx,nSamp);
					bool eqeq = ((equval & selector) == selector);
					bool vald = ((valval & selector) == selector);
					covA[threadIdx.x] += (vald && !eqeq && (P.pFlag[pairIndex] & P_ALPHA) == P_ALPHA) ? 1 : 0;
					covB[threadIdx.x] += (vald && eqeq && (P.pFlag[pairIndex] & P_ALPHA) == 0) ? 1 : 0;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//					if (threadIdx.x == 0 && threadIdx.y == 0)
//					if (threadIdx.x < 3 && intIdx == 1)
//						printf("\t%d\t%d,%d\tr:%d,c:%d  Pi %d  %d[%d] %x, T %d\n", intIdx, e1, e2, sIdy, sIdx, pairIndex, eqeq, vald, P.pFlag[pairIndex], threadIdx.x);
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
#endif

	// TODO: This could be done with an atomicAdd()
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int cA = 0;
		int cB = 0;
		int upp = min(blockDim.x * blockDim.y, nEP);
		for (int j = 0; j < upp; j++)
		{
			cA += covA[j];
			cB += covB[j];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//			printf("%d(%d,%d)   %d   +%d,%d\t+%d,%d\n", j, threadIdx.x, threadIdx.y, intIdx, covA[j], cA, covB[j], cB);
#endif
		}
		for (int j = 0; j < upp; j++)
		{
			covA[j] = cA;
			covB[j] = cB;
		}

		// At this point we have the

	}
	__syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
	if (threadIdx.x == 0 && threadIdx.y == 0)
		printf("Interaction:[%d,%d:%d] cA:%d cB:%d\n", idxA, idxB, intIdx, covA[threadIdx.x], covB[threadIdx.x]);
#endif

}
#undef	DEBUG_PRINTS


// Launcher of the kernel 1 function.
template <typename T>
__host__ static void launch_process_gpu_2(PlinkReader<T> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ppres, const struct _paramP1 &par)
{
	struct _dataPointersPairsD<T>	d_PPtrs;
	size_t	byteCounter;
	double	totTime = 0.0f;

	Timer tt;
	CUDACHECK(cudaSetDevice(par.gpuNum),par.gpuNum);
	byteCounter = allocSendData_device(*pr, d_PPtrs, nPairs, par);
	clog << "K2: InputData: " << byteCounter << std::endl;

	// Output buffer
	struct cudaDeviceProp dProp;
	std::memset(&dProp,0,sizeof(dProp));
	CUDACHECK(cudaGetDeviceProperties(&dProp,par.gpuNum),par.gpuNum);
	clog << "K2: Device: " << par.gpuNum << " [" << dProp.name << "] " << dProp.major << "." << dProp.minor << endl;
	size_t availMem, totalMem;
	CUDACHECK(cudaMemGetInfo(&availMem,&totalMem),par.gpuNum);
	int nPairsMax = availMem / (nResPP * sizeof(PairInteractionResult));
	clog << "K2: Max Global: " << dProp.totalGlobalMem << ", Available: " << (availMem+ONEMEGA-1)/ONEMEGA << "MB. Max nPairs: " << nPairsMax << endl;
	if (nPairsMax < nPairs)
	{
		clog << "K2: Available Memory is not enough for " << nPairs << " pairs with " << nResPP << " results each.  Aborting." << endl << endl;
		freeData_device(d_PPtrs, par);
		return;
	}

	PairInteractionResult	*d_pir, *h_pir;
	byteCounter += allocResults_device(&d_pir, nPairs, par);
	allocResults_host(&h_pir, nPairs, par);

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
	k2_g<T><<<grid,block,16384>>>(d_PPtrs, fgr, pr->numSNPs(), pr->numSamples(), pr->elementsSNP(), nPairs, nResPP, d_pir);

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

	freeData_device(d_PPtrs, par);
	freeResults_device(*d_pir, par);
	freeResults_host(*h_pir, par);

	return;
}


__host__ void process_gpu_2(PlinkReader<ui8> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ptops, const struct _paramP1 &par)
{
	launch_process_gpu_2<ui8>(pr, nPairs, nResPP, ptops, par);
}

__host__ void process_gpu_2(PlinkReader<ui4> *pr, const size_t nPairs, const int nResPP, PairInteractionResult *ptops, const struct _paramP1 &par)
{

}
