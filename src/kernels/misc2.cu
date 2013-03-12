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
#include "../proc_gpu.h"

// Allocate results buffers in device
// The struct ld_abkr is assumed to have the parameters completed previously by the
// heuristic used to dimension buffers on device based on available memory
int allocResults_device(const ABKResultDetails &ld_abkr, ABKResultDetails **p_abkr, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMalloc((void**)p_abkr,sizeof(ABKResultDetails)),par.gpuNum);
	CUDACHECK(cudaMalloc((void**)&ld_abkr.pairList,
			  ld_abkr.nPairs * ld_abkr.dResPP * sizeof(short int)),par.gpuNum);
	CUDACHECK(cudaMalloc((void**)&ld_abkr.selected,
			  ld_abkr.maxSelected * sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += sizeof(ABKResultDetails)
			+ ld_abkr.nPairs * ld_abkr.dResPP * sizeof(short int)
			+ ld_abkr.maxSelected * sizeof(PairInteractionResult);
	CUDACHECK(cudaMemcpy(*p_abkr, &ld_abkr, sizeof(ABKResultDetails),cudaMemcpyHostToDevice),par.gpuNum);
	return byteCounter;
}

int allocResults_host(ABKResultDetails &h_abkr, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMallocHost((void**)&h_abkr.pairList,
			  h_abkr.nPairs * h_abkr.dResPP * sizeof(short int)),par.gpuNum);
	CUDACHECK(cudaMalloc((void**)&h_abkr.selected,
			  h_abkr.maxSelected * sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += h_abkr.nPairs * h_abkr.dResPP * sizeof(short int)
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
	CUDACHECK(cudaFree((void*)h_abkr.pairList),par.gpuNum);
	CUDACHECK(cudaFree((void*)h_abkr.selected),par.gpuNum);
}

/*
 * Allocates and initialises the device buffers to hold results for
 * each invocation of the grid.
 * Returns number of bytes allocated in device or host (the same), or -1 if error.
 */
int allocResults_device(PairInteractionResult **d_pir, int nPIR, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMalloc((void**)d_pir,nPIR*sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += nPIR * sizeof(PairInteractionResult) + 1;


	return byteCounter;
}

int allocResults_host(PairInteractionResult **d_pir, int nPIR, const struct _paramP1 &par)
{
	int byteCounter = 0;
	CUDACHECK(cudaMallocHost((void**)d_pir,nPIR*sizeof(PairInteractionResult)),par.gpuNum);
	byteCounter += nPIR * sizeof(PairInteractionResult) + 1;


	return byteCounter;
}

void freeResults_device(const PairInteractionResult &pir, const struct _paramP1 &par)
{
	CUDACHECK(cudaFree((void*)&pir),par.gpuNum);
}

void freeResults_host(const PairInteractionResult &pir, const struct _paramP1 &par)
{
	CUDACHECK(cudaFreeHost((void*)&pir),par.gpuNum);
}
