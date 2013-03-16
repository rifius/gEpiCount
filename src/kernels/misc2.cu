/*
 * results_k2.cpp
 *
 * Contains functions auxiliary to management of results for kernel 2
 *
 *  Created on: 15/02/2013
 *      Author: carlos
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
#include <cuda_runtime.h>

#include "../../inc/gABKEpi.h"
#include "../ABKEpiGraph.h"
#include "../proc_gpu.h"

#define	DEBUG_PRINTS

// Allocate results buffers in device
// The struct ld_abkr is assumed to have the parameters completed previously by the
// heuristic used to dimension buffers on device based on available memory
int allocResults_device(const ABKResultDetails &ld_abkr, ABKResultDetails **p_abkr, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMalloc((void**)p_abkr,sizeof(ABKResultDetails)),par.gpuNum);
	CUDACHECK(cudaMalloc((void**)&ld_abkr.pairList,
			  ld_abkr.nPairs * ld_abkr.dResPP * sizeof(PILindex_t)),par.gpuNum);
	CUDACHECK(cudaMalloc((void**)&ld_abkr.selected,
			  ld_abkr.maxSelected * sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += sizeof(ABKResultDetails)
			+ ld_abkr.nPairs * ld_abkr.dResPP * sizeof(PILindex_t)
			+ ld_abkr.maxSelected * sizeof(PairInteractionResult);
	CUDACHECK(cudaMemcpy(*p_abkr, &ld_abkr, sizeof(ABKResultDetails),cudaMemcpyHostToDevice),par.gpuNum);
	clog << "ABKResultDetails " << *p_abkr << " PL: " << ld_abkr.pairList << " WC: " << ld_abkr.selected << endl;
	return byteCounter;
}

int allocResults_host(ABKResultDetails &h_abkr, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMallocHost((void**)&h_abkr.pairList,
			  h_abkr.nPairs * h_abkr.dResPP * sizeof(PILindex_t)),par.gpuNum);
	CUDACHECK(cudaMallocHost((void**)&h_abkr.selected,
			  h_abkr.maxSelected * sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += h_abkr.nPairs * h_abkr.dResPP * sizeof(PILindex_t)
				   + h_abkr.maxSelected * sizeof(PairInteractionResult);
	return byteCounter;
}

void freeResults_device(const ABKResultDetails &ld_abkr, const ABKResultDetails *p_abkr, const struct _paramP1 &par)
{
	CUDACHECK(cudaFree((void*)ld_abkr.pairList),par.gpuNum);
	CUDACHECK(cudaFree((void*)ld_abkr.selected),par.gpuNum);
	CUDACHECK(cudaFree((void*)p_abkr),par.gpuNum);
}

void freeResults_host(const ABKResultDetails &h_abkr, const struct _paramP1 &par)
{
	CUDACHECK(cudaFreeHost((void*)h_abkr.pairList),par.gpuNum);
	CUDACHECK(cudaFreeHost((void*)h_abkr.selected),par.gpuNum);
}

// Kernel to initialize the result list per pair.  To be launched as a linear grid of blocks.
__global__ void k2_initPairList(ABKResultDetails *pr)
{
	int step = gridDim.x * blockDim.x * blockDim.y;
	int idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int N = (pr->nPairs + step - 1) / step;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200) && defined(DEBUG_PRINTS)
//	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
//	{
//		printf("pr:%p->dResP:%d nPairs:%d maxSelected:%d pairList:%p selected:%p\n", pr, pr->dResPP, pr->nPairs, pr->maxSelected, pr->pairList, pr->selected);
//		printf("B(%d,%d) T(%d,%d) idx:%d N:%d step:%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, N, step);
//	}
#endif
	for (int j = 0; j < N; j++)
	{
		if (idx >= pr->nPairs)
			return;
		int pli = idx * pr->dResPP;
		for (int k = 0; k < pr->dResPP; k++)
		{
			pr->pairList[pli + k] = pr->maxSelected;
		}
		idx += step;
	}
	// This last bit here so we save one transfer.
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		pr->currIndex = 0;
}

bool transferBack(const ABKResultDetails *pd_abk, ABKResultDetails *h_abk, const struct _paramP1 &par)
{
	ABKResultDetails  mabk;
	CUDACHECK(cudaMemcpy(&mabk,pd_abk,sizeof(ABKResultDetails),cudaMemcpyDeviceToHost),par.gpuNum);
	CUDACHECK(cudaMemcpy(h_abk->selected,mabk.selected,mabk.maxSelected*sizeof(PairInteractionResult),cudaMemcpyDeviceToHost),par.gpuNum);
	CUDACHECK(cudaMemcpy(h_abk->pairList,mabk.pairList,mabk.nPairs*mabk.dResPP*sizeof(PILindex_t),cudaMemcpyDeviceToHost),par.gpuNum);
	h_abk->currIndex = mabk.currIndex;
}
